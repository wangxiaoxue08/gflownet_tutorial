# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "jax[cpu]==0.7.2",
#   "flax==0.11.2",
#   "optax==0.2.6",
#   "numpy==2.2.6",
# ]
# ///
"""Trajectory Balance GFlowNet: 从 5 个元素中无序地选 3 个。

运行（uv 自动安装上面锁定的 CPU 依赖）:
    uv run --script gflownet_nnx_tutorial.py
    uv run --script gflownet_nnx_tutorial.py --backward uniform

默认学习 P_F、P_B、log Z；uniform 模式固定 P_B 为均匀删除。
训练只评估采到的终态的 log R，不使用精确 Z 或枚举数据。
训练后枚举 60 条轨迹，验证 10 个终态的概率，再检验实际采样。
与配套 HTML 的函数名、代码块和符号对应。
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations, permutations
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

N_ITEMS = 5
SET_SIZE = 3


def log_reward(bits: jax.Array) -> jax.Array:
    """返回终态 log R；两个偏好的集合分别是 {0,1,2}、{2,3,4}。"""
    mode_a = jnp.array([1, 1, 1, 0, 0], dtype=jnp.float32)
    mode_b = jnp.array([0, 0, 1, 1, 1], dtype=jnp.float32)
    distance_a = 0.5 * jnp.abs(bits - mode_a).sum(axis=-1)
    distance_b = 0.5 * jnp.abs(bits - mode_b).sum(axis=-1)
    return jnp.logaddexp(-2.5 * distance_a, -2.5 * distance_b)


def masked_log_softmax(logits: jax.Array, legal: jax.Array) -> jax.Array:
    """只在合法动作上归一化；调用方保证每行至少一个合法动作。"""
    masked_logits = jnp.where(legal, logits, -jnp.inf)
    return jax.nn.log_softmax(masked_logits, axis=-1)


class PolicyMLP(nnx.Module):
    """输入 5 维 bit state，输出 5 个动作的未归一化 logits。"""

    def __init__(self, *, rngs: nnx.Rngs, width: int = 64):
        """建立两个隐藏层；nnx.Linear 的权重和偏置都是 nnx.Param。"""
        self.fc1 = nnx.Linear(N_ITEMS, width, rngs=rngs)
        self.fc2 = nnx.Linear(width, width, rngs=rngs)
        self.out = nnx.Linear(width, N_ITEMS, rngs=rngs)

    def __call__(self, state: jax.Array) -> jax.Array:
        """对单个 state 或 [batch, 5] states 计算 logits。"""
        h = jax.nn.silu(self.fc1(state))
        h = jax.nn.silu(self.fc2(h))
        return self.out(h)


class SetGFlowNet(nnx.Module):
    """分别参数化 P_F、可选的 P_B，以及标量 z = log Z。"""

    def __init__(self, *, rngs: nnx.Rngs, learn_backward: bool = True):
        """uniform 模式不建立反向网络，因此没有未使用的反向参数。"""
        self.forward = PolicyMLP(rngs=rngs)
        self.backward = PolicyMLP(rngs=rngs) if learn_backward else None
        self.log_z = nnx.Param(jnp.array(0.0, dtype=jnp.float32))

    def forward_log_probs(self, state: jax.Array) -> jax.Array:
        """前向添加一个未选元素；只在 |state| < 3 时调用。"""
        return masked_log_softmax(self.forward(state), state == 0)

    def backward_log_probs(self, child: jax.Array) -> jax.Array:
        """从 child 删除一个已有元素；不能在空集 source 上调用。"""
        logits = self.backward(child) if self.backward is not None else jnp.zeros_like(child)
        return masked_log_softmax(logits, child == 1)


@nnx.jit
def sample_actions(
    model: SetGFlowNet,
    key: jax.Array,
    initial_state: jax.Array,
    epsilon: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """从全零 source 批量采 3 步；epsilon 只控制收集数据时的探索。"""
    state = initial_state
    actions = []
    for _ in range(SET_SIZE):
        key, action_key = jax.random.split(key)
        log_pf = model.forward_log_probs(state)
        legal = (state == 0).astype(jnp.float32)
        uniform = legal / legal.sum(axis=-1, keepdims=True)
        behavior = (1.0 - epsilon) * jnp.exp(log_pf) + epsilon * uniform
        # categorical 接受 logits / log probabilities，不接受 probabilities。
        action = jax.random.categorical(action_key, jnp.log(behavior), axis=-1)
        actions.append(action)
        state = state.at[jnp.arange(state.shape[0]), action].set(1.0)
    return jnp.stack(actions, axis=1), state


@nnx.jit
def sample_backward(
    model: SetGFlowNet, key: jax.Array, terminal: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """从完整 3 元集合按 P_B 采删除序列；每行最终回到空集。"""
    state = terminal
    removals = []
    for _ in range(SET_SIZE):
        key, action_key = jax.random.split(key)
        action = jax.random.categorical(action_key, model.backward_log_probs(state))
        removals.append(action)
        state = state.at[jnp.arange(state.shape[0]), action].set(0.0)
    return jnp.stack(removals, axis=1), state


def trajectory_statistics(
    model: SetGFlowNet, actions: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """固定已采 actions 重放轨迹，累加当前 P_F 和 P_B 的 log 概率。"""
    batch_size = actions.shape[0]
    rows = jnp.arange(batch_size)
    state = jnp.zeros((batch_size, N_ITEMS), dtype=jnp.float32)
    log_pf = jnp.zeros(batch_size, dtype=jnp.float32)
    log_pb = jnp.zeros(batch_size, dtype=jnp.float32)
    for t in range(SET_SIZE):
        action = actions[:, t]
        log_pf += model.forward_log_probs(state)[rows, action]
        child = state.at[rows, action].set(1.0)
        # 必须在 child 上评估反向概率；逆动作是删除同一个 action。
        log_pb += model.backward_log_probs(child)[rows, action]
        state = child
    return log_pf, log_pb, log_reward(state), state


def tb_loss(model: SetGFlowNet, actions: jax.Array) -> jax.Array:
    """TB residual 平方的 batch 均值；reward 固定，actions 不参与微分。"""
    log_pf, log_pb, log_r, _ = trajectory_statistics(model, actions)
    residual = model.log_z.value + log_pf - log_r - log_pb
    return jnp.mean(jnp.square(residual))


@nnx.jit
def train_step(
    model: SetGFlowNet, optimizer: nnx.Optimizer, actions: jax.Array
) -> jax.Array:
    """只更新模型中 nnx.Param：前向/反向 MLP 和 log_z。"""
    loss, grads = nnx.value_and_grad(
        tb_loss, argnums=nnx.DiffState(0, nnx.Param)
    )(model, actions)
    optimizer.update(model, grads)
    return loss


def train(args: argparse.Namespace) -> tuple[SetGFlowNet, list[dict]]:
    """每轮先探索采轨迹，再把离散动作固定下来进行一次 TB 更新。"""
    model = SetGFlowNet(rngs=nnx.Rngs(args.seed), learn_backward=args.backward == "learned")
    optimizer = nnx.Optimizer(model, optax.adam(2e-3), wrt=nnx.Param)
    key = jax.random.key(args.seed + 17)
    initial = jnp.zeros((args.batch_size, N_ITEMS), dtype=jnp.float32)
    epsilon = jnp.asarray(0.2, dtype=jnp.float32)
    history = []
    for step in range(args.steps):
        key, sample_key = jax.random.split(key)
        actions, _ = sample_actions(model, sample_key, initial, epsilon)
        loss = train_step(model, optimizer, actions)
        if step % 500 == 0 or step == args.steps - 1:
            row = {"step": step + 1, "loss": float(loss), "log_z": float(model.log_z.value)}
            history.append(row)
            print(f"step={step + 1:4d} loss={row['loss']:.7f} logZ={row['log_z']:.6f}")
    return model, history


def enumerate_terminals() -> tuple[list[tuple[int, ...]], np.ndarray]:
    """生成 10 个终态；仅供训练完成后的精确诊断。"""
    subsets = list(combinations(range(N_ITEMS), SET_SIZE))
    bits = np.zeros((len(subsets), N_ITEMS), dtype=np.float32)
    for i, ids in enumerate(subsets):
        bits[i, list(ids)] = 1.0
    return subsets, bits


def exact_diagnostics(model: SetGFlowNet) -> dict:
    """枚举所有 60 条轨迹，区分真实模型误差和 Monte Carlo 采样误差。"""
    subsets, bits = enumerate_terminals()
    rewards = np.exp(np.asarray(log_reward(jnp.asarray(bits)), dtype=np.float64))
    paths = jnp.asarray(list(permutations(range(N_ITEMS), SET_SIZE)), dtype=jnp.int32)
    log_pf, log_pb, log_r, terminal = nnx.jit(trajectory_statistics)(model, paths)
    pf, pb = np.exp(np.asarray(log_pf, dtype=np.float64)), np.exp(np.asarray(log_pb, dtype=np.float64))
    group = np.all(np.asarray(terminal)[:, None, :] == bits[None, :, :], axis=-1)
    model_p, backward_sums = pf @ group, pb @ group
    residual = float(model.log_z.value) + np.asarray(log_pf) - np.asarray(log_r) - np.asarray(log_pb)
    np.testing.assert_allclose(model_p.sum(), 1.0, atol=2e-6)
    np.testing.assert_allclose(backward_sums, 1.0, atol=2e-6)
    assert np.all(np.isfinite(residual)), "TB residual 必须有限"
    return {
        "subsets": [list(s) for s in subsets], "rewards": rewards.tolist(),
        "target": (rewards / rewards.sum()).tolist(), "model": model_p.tolist(),
        "log_z_exact": float(np.log(rewards.sum())), "log_z_learned": float(model.log_z.value),
        "tv_exact": float(0.5 * np.abs(model_p - rewards / rewards.sum()).sum()),
        "tb_rmse_all_paths": float(np.sqrt(np.mean(residual**2))),
        "tb_max_abs_all_paths": float(np.max(np.abs(residual))),
        "forward_path_sum": float(pf.sum()),
        "backward_path_sums": backward_sums.tolist(),
    }


def sample_diagnostics(model: SetGFlowNet, n: int, seed: int) -> np.ndarray:
    """用 epsilon=0 从学得的 P_F 直接采样；返回 10 个终态的经验频率。"""
    initial = jnp.zeros((n, N_ITEMS), dtype=jnp.float32)
    actions, terminal = sample_actions(model, jax.random.key(seed), initial, jnp.array(0.0))
    actions, terminal = np.asarray(actions), np.asarray(terminal)
    assert np.all(np.diff(np.sort(actions, axis=1), axis=1) > 0), "同一轨迹不能重复添加元素"
    assert np.all(terminal.sum(axis=-1) == SET_SIZE), "采样必须在 3 元集合处结束"
    _, bits = enumerate_terminals()
    return np.all(terminal[:, None, :] == bits[None, :, :], axis=-1).mean(axis=0)


def evaluate(model: SetGFlowNet, args: argparse.Namespace, history: list[dict]) -> dict:
    """打印终态对照表，并返回可序列化的完整诊断结果。"""
    result = exact_diagnostics(model)
    empirical = sample_diagnostics(model, args.samples, args.seed + 2026)
    result.update({"empirical": empirical.tolist(), "history": history,
                   "backward": args.backward, "steps": args.steps, "seed": args.seed,
                   "batch_size": args.batch_size, "samples": args.samples})
    result["tv_empirical"] = float(0.5 * np.abs(empirical - result["target"]).sum())
    result["versions"] = {"jax": jax.__version__, "flax": flax.__version__,
                          "optax": optax.__version__, "numpy": np.__version__}
    print("\nterminal      target     model(exact)  sampled")
    for s, target, predicted, observed in zip(result["subsets"], result["target"], result["model"], empirical):
        print(f"{str(tuple(s)):12s}  {target:.6f}   {predicted:.6f}      {observed:.6f}")
    print(f"\nExact model TV = {result['tv_exact']:.7f}")
    print(f"Empirical TV   = {result['tv_empirical']:.7f}  (n={args.samples})")
    print(f"log Z learned  = {result['log_z_learned']:.7f}; exact = {result['log_z_exact']:.7f}")
    print(f"All-path TB RMSE = {result['tb_rmse_all_paths']:.7f}")
    return result


def parse_args() -> argparse.Namespace:
    """读取可复现实验配置；默认学习反向策略。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backward", choices=["learned", "uniform"], default="learned")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-json", type=Path, help="可选：把训练日志和诊断保存为 JSON")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.samples) <= 0 or args.seed < 0:
        parser.error("steps、batch-size、samples 必须为正数；seed 必须非负")
    return args


def main() -> None:
    """训练，再独立验证；只有指定 --report-json 才额外写出结果文件。"""
    args = parse_args()
    print(f"JAX {jax.__version__} | Flax {flax.__version__} | {jax.devices()}")
    model, history = train(args)
    report = evaluate(model, args, history)
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
