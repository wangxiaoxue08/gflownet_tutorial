"""Trajectory Balance GFlowNet: 从 5 个元素中无序地选 3 个。

运行（使用项目 uv.lock 和 .venv）::

    uv run python gflownet_nnx_tutorial.py
    uv run python gflownet_nnx_tutorial.py --backward uniform
    uv run python gflownet_nnx_tutorial.py --checkpoint runs/<run>/model.msgpack

默认学习 P_F、P_B、log Z；``--backward uniform`` 固定 P_B 为均匀删除。
训练只评估采到的终态的 log R，不使用精确 Z 或枚举数据。
训练后枚举 60 条轨迹，验证 10 个终态的概率，再检验实际采样。
与配套 HTML 的函数名、代码块和符号对应。

本文件只保留模型、采样、TB loss 与训练；CLI / 诊断 / checkpoint 在 utility。
"""

from __future__ import annotations

import argparse
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from utility import (
    N_ITEMS,
    SET_SIZE,
    enumerate_terminals,
    evaluate,
    exact_diagnostics,
    generate_samples,
    load_model,
    parse_args,
    run_experiment,
    sample_diagnostics,
    save_model,
    summarize_samples,
)

# Public re-exports used by tests and callers of this tutorial module.
__all__ = [
    "N_ITEMS",
    "SET_SIZE",
    "PolicyMLP",
    "SetGFlowNet",
    "enumerate_terminals",
    "evaluate",
    "exact_diagnostics",
    "generate_samples",
    "load_model",
    "log_reward",
    "masked_log_softmax",
    "parse_args",
    "sample_actions",
    "sample_backward",
    "sample_diagnostics",
    "save_model",
    "summarize_samples",
    "tb_loss",
    "train",
    "train_step",
    "trajectory_statistics",
]


def log_reward(bits: jax.Array) -> jax.Array:
    """计算终态的 log 奖励 ``log R(s)``。

    奖励在两个偏好集合上峰值：``{0,1,2}`` 与 ``{2,3,4}``。对每个 mode，
    用对称差大小的一半作距离，再对两个 Boltzmann 权重取 ``logaddexp``。

    Args:
        bits: float32，形状 ``[..., 5]``，取值为 0/1，表示已选元素。

    Returns:
        float32，形状 ``[...]``，每个终态的 ``log R``。

    Examples:
        恰好等于 mode A ``{0,1,2}`` 时距离为 0，奖励接近 1（log ≈ 0）::

            >>> float(log_reward(jnp.array([1.0, 1.0, 1.0, 0.0, 0.0])))
            0.006715...  # logaddexp(0, -5)

        批量输入保持前缀维度::

            >>> log_reward(jnp.zeros((2, 5))).shape
            (2,)
    """
    mode_a = jnp.array([1, 1, 1, 0, 0], dtype=jnp.float32)
    mode_b = jnp.array([0, 0, 1, 1, 1], dtype=jnp.float32)
    distance_a = 0.5 * jnp.abs(bits - mode_a).sum(axis=-1)
    distance_b = 0.5 * jnp.abs(bits - mode_b).sum(axis=-1)
    return jnp.logaddexp(-2.5 * distance_a, -2.5 * distance_b)


def masked_log_softmax(logits: jax.Array, legal: jax.Array) -> jax.Array:
    """只在合法动作上做 log-softmax；非法位置为 ``-inf``。

    调用方须保证每行至少一个合法动作，否则该行会得到 NaN。

    Args:
        logits: float32，形状 ``[..., 5]``，未归一化动作分数。
        legal: 与 ``logits`` 同形状的布尔/0-1 mask；True/1 表示合法。

    Returns:
        float32，形状 ``[..., 5]``；合法位置上的 log 概率，非法位置为 ``-inf``。

    Examples:
        空集上可添加任意元素，log 概率均匀为 ``log(1/5)``::

            >>> masked_log_softmax(jnp.zeros(5), jnp.ones(5, dtype=bool))
            Array([-1.609438, ...], dtype=float32)

        已选 ``{0,1}`` 时只能添加 2/3/4::

            >>> state = jnp.array([1.0, 1.0, 0.0, 0.0, 0.0])
            >>> masked_log_softmax(jnp.zeros(5), state == 0)
            Array([-inf, -inf, -1.0986, -1.0986, -1.0986], dtype=float32)
    """
    masked_logits = jnp.where(legal, logits, -jnp.inf)
    return jax.nn.log_softmax(masked_logits, axis=-1)


class PolicyMLP(nnx.Module):
    """两层 SiLU MLP：5 维 bit state → 5 个动作 logits。"""

    def __init__(self, *, rngs: nnx.Rngs, width: int = 64):
        """建立两个隐藏层；``nnx.Linear`` 的权重和偏置都是 ``nnx.Param``。

        Args:
            rngs: NNX 随机流，用于初始化线性层。
            width: 隐藏层宽度，默认 64。
        """
        self.fc1 = nnx.Linear(N_ITEMS, width, rngs=rngs)
        self.fc2 = nnx.Linear(width, width, rngs=rngs)
        self.out = nnx.Linear(width, N_ITEMS, rngs=rngs)

    def __call__(self, state: jax.Array) -> jax.Array:
        """对单个或批量 state 计算未归一化 logits。

        Args:
            state: float32，形状 ``[5]`` 或 ``[batch, 5]``。

        Returns:
            float32，形状与 ``state`` 相同前缀、末维为 5 的 logits。
        """
        h = jax.nn.silu(self.fc1(state))
        h = jax.nn.silu(self.fc2(h))
        return self.out(h)


class SetGFlowNet(nnx.Module):
    """分别参数化前向策略 P_F、可选反向策略 P_B，以及标量 ``z = log Z``。"""

    def __init__(self, *, rngs: nnx.Rngs, learn_backward: bool = True, width: int = 64):
        """构造网络。

        ``learn_backward=False``（CLI 的 uniform 模式）不建立反向 MLP，
        因此没有未使用的反向参数；此时 ``backward_log_probs`` 退化为均匀删除。

        Args:
            rngs: NNX 随机流。
            learn_backward: 是否学习 P_B；False 时固定均匀反向。
            width: 策略 MLP 隐藏宽度。
        """
        self.width = width
        self.forward = PolicyMLP(rngs=rngs, width=width)
        self.backward = PolicyMLP(rngs=rngs, width=width) if learn_backward else None
        self.log_z = nnx.Param(jnp.array(0.0, dtype=jnp.float32))

    def forward_log_probs(self, state: jax.Array) -> jax.Array:
        """前向：从 ``state`` 添加一个尚未选中的元素。

        只应在 ``|state| < SET_SIZE`` 时调用。

        Args:
            state: float32，``[..., 5]``，当前已选集的 bit 向量。

        Returns:
            float32，``[..., 5]``，在 ``state == 0`` 位置上归一化的 ``log P_F``。

        Examples:
            空集上 5 个添加动作合法；部分填满后只对未选位置归一化::

                >>> model = SetGFlowNet(rngs=nnx.Rngs(0), width=8)
                >>> model.forward_log_probs(jnp.zeros(5)).shape
                (5,)
        """
        return masked_log_softmax(self.forward(state), state == 0)

    def backward_log_probs(self, child: jax.Array) -> jax.Array:
        """反向：从 ``child`` 删除一个已有元素。

        不能在空集 source 上调用。无反向网络时，对已选元素给出均匀 ``log P_B``。

        Args:
            child: float32，``[..., 5]``，非空 bit 向量。

        Returns:
            float32，``[..., 5]``，在 ``child == 1`` 位置上归一化的 ``log P_B``。
        """
        logits = (
            self.backward(child) if self.backward is not None else jnp.zeros_like(child)
        )
        return masked_log_softmax(logits, child == 1)


@nnx.jit
def sample_actions(
    model: SetGFlowNet,
    key: jax.Array,
    initial_state: jax.Array,
    epsilon: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """按 ε-greedy 行为策略从空集采 ``SET_SIZE`` 步添加动作。

    ``epsilon`` 只控制训练探索；评估应传入 0。每一步使用独立子 key。
    ``categorical`` 需要 logits，因此对行为概率取 ``log``。

    Args:
        model: 当前 GFlowNet（读 ``P_F``）。
        key: JAX PRNG key。
        initial_state: float32，``[batch, 5]``，通常全 0。
        epsilon: float32 标量，探索率 ∈ [0, 1]。

    Returns:
        ``(actions, terminal)``：

        - ``actions``: int32，``[batch, 3]``，添加顺序（可含重复则非法）。
        - ``terminal``: float32，``[batch, 5]``，对应的 3-元终态 bit 向量。

    Examples:
        无探索时采样一批轨迹::

            >>> model = SetGFlowNet(rngs=nnx.Rngs(0), width=8)
            >>> init = jnp.zeros((4, 5))
            >>> actions, terminal = sample_actions(
            ...     model, jax.random.key(0), init, jnp.array(0.0)
            ... )
            >>> actions.shape, terminal.shape
            ((4, 3), (4, 5))
    """
    state = initial_state
    actions = []
    for _ in range(SET_SIZE):
        key, action_key = jax.random.split(key)
        log_pf = model.forward_log_probs(state)
        legal = (state == 0).astype(jnp.float32)
        uniform = legal / legal.sum(axis=-1, keepdims=True)
        behavior = (1.0 - epsilon) * jnp.exp(log_pf) + epsilon * uniform
        action = jax.random.categorical(action_key, jnp.log(behavior), axis=-1)
        actions.append(action)
        state = state.at[jnp.arange(state.shape[0]), action].set(1.0)
    return jnp.stack(actions, axis=1), state


@nnx.jit
def sample_backward(
    model: SetGFlowNet, key: jax.Array, terminal: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """从完整 3 元集合按 ``P_B`` 采删除序列，直到回到空集。

    Args:
        model: 当前 GFlowNet（读 ``P_B`` 或均匀反向）。
        key: JAX PRNG key。
        terminal: float32，``[batch, 5]``，每个样本恰有 3 个 1。

    Returns:
        ``(removals, source)``：

        - ``removals``: int32，``[batch, 3]``，删除顺序。
        - ``source``: float32，``[batch, 5]``，应为全 0。

    Examples:
        ::

            >>> model = SetGFlowNet(rngs=nnx.Rngs(0), width=8)
            >>> terminal = jnp.array([[1.0, 1.0, 1.0, 0.0, 0.0]])
            >>> removals, source = sample_backward(model, jax.random.key(1), terminal)
            >>> removals.shape, float(source.sum())
            ((1, 3), 0.0)
    """
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
    """固定已采 ``actions`` 重放轨迹，累加当前 ``P_F`` / ``P_B`` 的 log 概率。

    反向概率必须在添加后的 child 上评估；逆动作是删除同一个元素。

    Args:
        model: 当前 GFlowNet。
        actions: int32，``[batch, 3]``，每行是一条添加顺序。

    Returns:
        ``(log_pf, log_pb, log_r, terminal)``，均为 batch 向量（末项为 ``[batch, 5]``）：

        - ``log_pf``: 轨迹前向 log 概率之和。
        - ``log_pb``: 轨迹反向 log 概率之和。
        - ``log_r``: 终态 ``log R``。
        - ``terminal``: 重放得到的终态 bit 向量。

    Examples:
        单条路径 ``0→1→2``::

            >>> model = SetGFlowNet(rngs=nnx.Rngs(0), width=8)
            >>> log_pf, log_pb, log_r, terminal = trajectory_statistics(
            ...     model, jnp.array([[0, 1, 2]], dtype=jnp.int32)
            ... )
            >>> terminal
            Array([[1., 1., 1., 0., 0.]], dtype=float32)
    """
    batch_size = actions.shape[0]
    rows = jnp.arange(batch_size)
    state = jnp.zeros((batch_size, N_ITEMS), dtype=jnp.float32)
    log_pf = jnp.zeros(batch_size, dtype=jnp.float32)
    log_pb = jnp.zeros(batch_size, dtype=jnp.float32)
    for t in range(SET_SIZE):
        action = actions[:, t]
        log_pf += model.forward_log_probs(state)[rows, action]
        child = state.at[rows, action].set(1.0)
        log_pb += model.backward_log_probs(child)[rows, action]
        state = child
    return log_pf, log_pb, log_reward(state), state


def tb_loss(model: SetGFlowNet, actions: jax.Array) -> jax.Array:
    """Trajectory Balance residual 平方的 batch 均值。

    目标：``log Z + log P_F(τ) ≈ log R(x) + log P_B(τ|x)``。
    ``actions`` 与 reward 视为常数，不对离散采样路径反传。

    Args:
        model: 含 ``log_z``、``P_F``、``P_B`` 的网络。
        actions: int32，``[batch, 3]``，固定的采样轨迹。

    Returns:
        float32 标量，``mean((log Z + log P_F - log R - log P_B)^2)``。
    """
    log_pf, log_pb, log_r, _ = trajectory_statistics(model, actions)
    residual = model.log_z[...] + log_pf - log_r - log_pb
    return jnp.mean(jnp.square(residual))


@nnx.jit
def train_step(
    model: SetGFlowNet, optimizer: nnx.Optimizer, actions: jax.Array
) -> jax.Array:
    """对一批固定轨迹做一次 TB 梯度更新。

    只更新 ``nnx.Param``：前向/反向 MLP 与 ``log_z``。

    Args:
        model: 可训练的 ``SetGFlowNet``（原地更新）。
        optimizer: 绑定到 ``nnx.Param`` 的 Optax 优化器包装。
        actions: int32，``[batch, 3]``。

    Returns:
        float32 标量，更新前的 TB loss。
    """
    loss, grads = nnx.value_and_grad(tb_loss, argnums=nnx.DiffState(0, nnx.Param))(
        model, actions
    )
    optimizer.update(model, grads)
    return loss


def train(args: argparse.Namespace) -> tuple[SetGFlowNet, list[dict]]:
    """每轮先探索采样轨迹，再固定离散动作做一次 TB 更新。

    Args:
        args: CLI 命名空间；使用 ``seed``、``backward``、``width``、
            ``learning_rate``、``batch_size``、``epsilon``、``steps``、``log_every``。

    Returns:
        ``(model, history)``：

        - ``model``: eval 模式下的训练后网络。
        - ``history``: 日志行列表，每项含 ``step`` / ``loss`` / ``log_z``。

    Raises:
        FloatingPointError: 任一记录的 loss 或 log Z 非有限。
    """
    model = SetGFlowNet(
        rngs=nnx.Rngs(args.seed),
        learn_backward=args.backward == "learned",
        width=args.width,
    )
    model.train()
    optimizer = nnx.Optimizer(model, optax.adam(args.learning_rate), wrt=nnx.Param)
    key = jax.random.key(args.seed + 17)
    initial = jnp.zeros((args.batch_size, N_ITEMS), dtype=jnp.float32)
    epsilon = jnp.asarray(args.epsilon, dtype=jnp.float32)
    history: list[dict] = []
    for step in range(args.steps):
        key, sample_key = jax.random.split(key)
        actions, _ = sample_actions(model, sample_key, initial, epsilon)
        loss = train_step(model, optimizer, actions)
        if step % args.log_every == 0 or step == args.steps - 1:
            row = {
                "step": step + 1,
                "loss": float(loss),
                "log_z": float(model.log_z[...]),
            }
            if not np.isfinite([row["loss"], row["log_z"]]).all():
                raise FloatingPointError(
                    f"Non-finite training state at step {step + 1}"
                )
            history.append(row)
            print(
                f"step={step + 1:4d} loss={row['loss']:.7f} logZ={row['log_z']:.6f}",
                flush=True,
            )
    model.eval()
    return model, history


# Bind tutorial numerics into utility helpers so utility never imports this
# module at runtime (avoids circular imports / double model copies).
exact_diagnostics = partial(
    exact_diagnostics,
    log_reward=log_reward,
    trajectory_statistics=trajectory_statistics,
)
generate_samples = partial(generate_samples, sample_actions=sample_actions)
sample_diagnostics = partial(sample_diagnostics, generate_samples=generate_samples)
load_model = partial(load_model, model_type=SetGFlowNet)
evaluate = partial(
    evaluate,
    exact_diagnostics=exact_diagnostics,
    generate_samples=generate_samples,
)


def main() -> None:
    """按 CLI 配置训练或加载 checkpoint，并写出实验报告。"""
    args = parse_args()
    if args.platform != "auto":
        jax.config.update("jax_platforms", args.platform)
    run_experiment(
        args,
        train=train,
        load_model=load_model,
        exact_diagnostics=exact_diagnostics,
        generate_samples=generate_samples,
        evaluate=evaluate,
    )


if __name__ == "__main__":
    main()
