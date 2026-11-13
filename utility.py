"""CLI, diagnostics, checkpoint I/O and experiment artifacts for the GFlowNet.

Core numerical functions and the model constructor are supplied by the tutorial.
This keeps utility imports independent of the executable module and avoids circular
imports or loading a second copy of the model when the tutorial runs as a script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from itertools import combinations, permutations
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx, serialization

if TYPE_CHECKING:
    from gflownet_nnx_tutorial import SetGFlowNet

N_ITEMS = 5
SET_SIZE = 3


def enumerate_terminals() -> tuple[list[tuple[int, ...]], np.ndarray]:
    """生成 10 个终态；仅供训练完成后的精确诊断。"""
    subsets = list(combinations(range(N_ITEMS), SET_SIZE))
    bits = np.zeros((len(subsets), N_ITEMS), dtype=np.float32)
    for i, ids in enumerate(subsets):
        bits[i, list(ids)] = 1.0
    return subsets, bits


def exact_diagnostics(
    model: SetGFlowNet, *, log_reward: Callable, trajectory_statistics: Callable
) -> dict:
    """枚举所有 60 条轨迹，区分真实模型误差和 Monte Carlo 采样误差。"""
    subsets, bits = enumerate_terminals()
    rewards = np.exp(np.asarray(log_reward(jnp.asarray(bits)), dtype=np.float64))
    paths = jnp.asarray(list(permutations(range(N_ITEMS), SET_SIZE)), dtype=jnp.int32)
    log_pf, log_pb, log_r, terminal = nnx.jit(trajectory_statistics)(model, paths)
    pf, pb = (
        np.exp(np.asarray(log_pf, dtype=np.float64)),
        np.exp(np.asarray(log_pb, dtype=np.float64)),
    )
    group = np.all(np.asarray(terminal)[:, None, :] == bits[None, :, :], axis=-1)
    model_p, backward_sums = pf @ group, pb @ group
    residual = (
        float(model.log_z[...])
        + np.asarray(log_pf)
        - np.asarray(log_r)
        - np.asarray(log_pb)
    )
    np.testing.assert_allclose(model_p.sum(), 1.0, atol=2e-6)
    np.testing.assert_allclose(backward_sums, 1.0, atol=2e-6)
    assert np.all(np.isfinite(residual)), "TB residual 必须有限"
    return {
        "subsets": [list(s) for s in subsets],
        "rewards": rewards.tolist(),
        "target": (rewards / rewards.sum()).tolist(),
        "model": model_p.tolist(),
        "log_z_exact": float(np.log(rewards.sum())),
        "log_z_learned": float(model.log_z[...]),
        "tv_exact": float(0.5 * np.abs(model_p - rewards / rewards.sum()).sum()),
        "tb_rmse_all_paths": float(np.sqrt(np.mean(residual**2))),
        "tb_max_abs_all_paths": float(np.max(np.abs(residual))),
        "forward_path_sum": float(pf.sum()),
        "backward_path_sums": backward_sums.tolist(),
    }


def generate_samples(
    model: SetGFlowNet,
    n: int,
    seed: int,
    batch_size: int = 1024,
    *,
    sample_actions: Callable,
) -> np.ndarray:
    """Return actual int32 addition orders [N, 3], using bounded CPU/device batches.

    Sampling uses P_F with epsilon=0. Seed and batch size jointly determine draws.
    Final batch padding is discarded; each batch receives an independent key.
    """
    if n <= 0 or batch_size <= 0 or seed < 0:
        raise ValueError("n/batch_size must be positive; seed must be non-negative")
    model.eval()
    batch_size = min(n, batch_size)
    initial = jnp.zeros((batch_size, N_ITEMS), dtype=jnp.float32)
    key = jax.random.key(seed)
    batches = []
    for start in range(0, n, batch_size):
        key, sample_key = jax.random.split(key)
        actions, terminal = sample_actions(model, sample_key, initial, jnp.array(0.0))
        host_actions, host_terminal = np.asarray(actions), np.asarray(terminal)
        if not np.all(np.diff(np.sort(host_actions, axis=1), axis=1) > 0):
            raise ValueError("A generated trajectory repeated an element")
        if not np.all(host_terminal.sum(axis=-1) == SET_SIZE):
            raise ValueError("A generated terminal did not contain three elements")
        batches.append(host_actions[: min(batch_size, n - start)])
    return np.concatenate(batches)


def summarize_samples(actions: np.ndarray) -> dict:
    """Count unordered sets from actual [N, 3] sampled trajectories, not a resample."""
    if actions.ndim != 2 or actions.shape[1] != SET_SIZE or not len(actions):
        raise ValueError("Expected a nonempty [N, 3] array of actions")
    sorted_sets = np.sort(actions, axis=1)
    if (
        not np.issubdtype(actions.dtype, np.integer)
        or np.any(actions < 0)
        or np.any(actions >= N_ITEMS)
        or np.any(np.diff(sorted_sets, axis=1) <= 0)
    ):
        raise ValueError("Samples must contain three distinct integer items in [0, 5)")
    subsets, _ = enumerate_terminals()
    matches = np.all(
        sorted_sets[:, None, :] == np.asarray(subsets)[None, :, :], axis=-1
    )
    counts = matches.sum(axis=0)
    empirical = counts / len(actions)
    # Marginal Wilson intervals: each count is binomial, not a simultaneous band.
    z = 1.959963984540054
    denominator = 1 + z**2 / len(actions)
    center = (empirical + z**2 / (2 * len(actions))) / denominator
    half = (
        z
        * np.sqrt(
            empirical * (1 - empirical) / len(actions) + z**2 / (4 * len(actions) ** 2)
        )
        / denominator
    )
    return {
        "counts": counts.tolist(),
        "empirical": empirical.tolist(),
        "interval_low": (center - half).tolist(),
        "interval_high": (center + half).tolist(),
        "examples": [
            {"index": i + 1, "actions": row.tolist(), "set": sorted_sets[i].tolist()}
            for i, row in enumerate(actions[:20])
        ],
    }


def sample_diagnostics(
    model: SetGFlowNet, n: int, seed: int, *, generate_samples: Callable
) -> np.ndarray:
    """Use epsilon=0 and return the ten empirical terminal frequencies."""
    return np.asarray(summarize_samples(generate_samples(model, n, seed))["empirical"])


def save_model(
    model: SetGFlowNet, path: Path, config: dict, history: list[dict]
) -> None:
    """Save all model variables and architecture for inference (not optimizer resume).

    The checkpoint contains no pickled executable objects. Existing files are refused.
    """
    payload = {
        "format_version": 1,
        "model_config": {
            "width": model.width,
            "learn_backward": model.backward is not None,
        },
        "training_config": config,
        "history": history,
        "state": nnx.to_pure_dict(nnx.state(model)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(serialization.msgpack_serialize(payload))


def load_model(
    path: Path, *, model_type: type[SetGFlowNet]
) -> tuple[SetGFlowNet, dict]:
    """Restore float32 model variables on the selected JAX backend and enter eval mode."""
    payload = serialization.msgpack_restore(path.read_bytes())
    if payload["format_version"] != 1:
        raise ValueError("Unsupported model checkpoint format")
    model = model_type(rngs=nnx.Rngs(0), **payload["model_config"])
    state = nnx.state(model)
    template = nnx.to_pure_dict(state)
    saved = payload.pop("state")
    if jax.tree.structure(template) != jax.tree.structure(saved):
        raise ValueError("Checkpoint state does not match the model architecture")
    for expected, actual in zip(
        jax.tree.leaves(template), jax.tree.leaves(saved), strict=True
    ):
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise ValueError("Checkpoint array shape or dtype mismatch")
        if not np.isfinite(actual).all():
            raise ValueError("Checkpoint contains non-finite model variables")
    nnx.replace_by_pure_dict(state, jax.tree.map(jnp.asarray, saved))
    nnx.update(model, state)
    model.eval()
    return model, payload


def evaluate(
    model: SetGFlowNet,
    args: argparse.Namespace,
    history: list[dict],
    actions: np.ndarray | None = None,
    *,
    exact_diagnostics: Callable,
    generate_samples: Callable,
) -> dict:
    """打印终态对照表，并返回可序列化的完整诊断结果。"""
    result = exact_diagnostics(model)
    if actions is None:
        actions = generate_samples(
            model, args.samples, args.sample_seed, args.sample_batch_size
        )
    if len(actions) != args.samples:
        raise ValueError("Sample count does not match the evaluation configuration")
    result.update(summarize_samples(actions))
    empirical = np.asarray(result["empirical"])
    result.update(
        {
            "history": history,
            "backward": args.backward,
            "steps": args.steps,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "samples": args.samples,
        }
    )
    result["tv_empirical"] = float(0.5 * np.abs(empirical - result["target"]).sum())
    result["tv_sampling"] = float(0.5 * np.abs(empirical - result["model"]).sum())
    result["versions"] = {
        "jax": jax.__version__,
        "flax": flax.__version__,
        "optax": optax.__version__,
        "numpy": np.__version__,
    }
    print("\nGenerated examples (addition order -> unordered set):")
    for row in result["examples"]:
        print(f"{row['index']:2d}. {row['actions']} -> {row['set']}")
    print("\nterminal      target     model(exact)  sampled")
    for s, target, predicted, observed in zip(
        result["subsets"], result["target"], result["model"], empirical
    ):
        print(f"{tuple(s)!s:12s}  {target:.6f}   {predicted:.6f}      {observed:.6f}")
    print(f"\nExact model TV = {result['tv_exact']:.7f}")
    print(f"Empirical TV   = {result['tv_empirical']:.7f}  (n={args.samples})")
    print(
        f"log Z learned  = {result['log_z_learned']:.7f}; exact = {result['log_z_exact']:.7f}"
    )
    print(f"All-path TB RMSE = {result['tb_rmse_all_paths']:.7f}")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """读取可复现实验配置；默认学习反向策略。"""
    parser = argparse.ArgumentParser(
        description="Trajectory Balance GFlowNet: 从 5 个元素中无序地选 3 个。"
    )
    parser.add_argument("--backward", choices=["learned", "uniform"], default="learned")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--sample-batch-size", type=int, default=1024)
    parser.add_argument("--platform", choices=["cpu", "auto"], default="cpu")
    parser.add_argument(
        "--output-dir", type=Path, help="New run directory (must not exist)"
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="Load a saved model; skip training"
    )
    parser.add_argument(
        "--report-json", type=Path, help="可选：把训练日志和诊断保存为 JSON"
    )
    parser.add_argument(
        "--publish-report",
        type=Path,
        help="Also copy the standalone HTML report to this path",
    )
    args = parser.parse_args(argv)
    if (
        min(
            args.steps,
            args.batch_size,
            args.samples,
            args.width,
            args.log_every,
            args.sample_batch_size,
        )
        <= 0
    ):
        parser.error(
            "steps, batch-size, samples, width, log-every, sample-batch-size must be positive"
        )
    if min(args.seed, args.sample_seed) < 0:
        parser.error("Seeds must be non-negative")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    if not 0 <= args.epsilon <= 1:
        parser.error("epsilon must be in [0, 1]")
    return args


def run_experiment(
    args: argparse.Namespace,
    *,
    train: Callable,
    load_model: Callable,
    exact_diagnostics: Callable,
    generate_samples: Callable,
    evaluate: Callable,
) -> None:
    """Train/save/reload/sample, or sample an existing checkpoint; write a unique report."""
    from experiment_report import write_report

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output = args.output_dir or Path("runs") / run_id
    output.mkdir(parents=True, exist_ok=False)
    print(f"Output: {output.resolve()}", flush=True)
    print(
        f"JAX {jax.__version__} | Flax {flax.__version__} | {jax.devices()}", flush=True
    )
    started = time.perf_counter()
    checkpoint = output / "model.msgpack"
    if args.checkpoint is None:
        model, history = train(args)
        config = {
            k: getattr(args, k)
            for k in (
                "steps",
                "batch_size",
                "seed",
                "width",
                "learning_rate",
                "epsilon",
                "backward",
                "log_every",
            )
        }
        jax.block_until_ready(nnx.state(model))
        training_seconds = time.perf_counter() - started
        before = np.asarray(exact_diagnostics(model)["model"])
        save_model(model, checkpoint, config, history)
        model, metadata = load_model(checkpoint)
        after = np.asarray(exact_diagnostics(model)["model"])
        np.testing.assert_array_equal(before, after)
        restore_error = float(np.max(np.abs(before - after)))
    else:
        model, metadata = load_model(args.checkpoint)
        config, history = metadata["training_config"], metadata["history"]
        save_model(model, checkpoint, config, history)
        training_seconds, restore_error = None, None
        for name, value in config.items():
            setattr(args, name, value)
    actions = generate_samples(
        model, args.samples, args.sample_seed, args.sample_batch_size
    )
    report = evaluate(model, args, history, actions)
    report.update(
        {
            "run_id": run_id,
            "created_utc": datetime.now(UTC).isoformat(),
            "training_config": config,
            "sampling_config": {
                "seed": args.sample_seed,
                "batch_size": args.sample_batch_size,
                "n": args.samples,
                "epsilon": 0.0,
            },
            "training_seconds_including_compile": training_seconds,
            "restore_max_probability_error": restore_error,
            "source_checkpoint": str(args.checkpoint.resolve())
            if args.checkpoint
            else None,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "hardware": {
                "os": platform.platform(),
                "processor": platform.processor(),
                "python": sys.version,
                "devices": [str(d) for d in jax.devices()],
                "backend": jax.default_backend(),
            },
            "task": "5-choose-3-v1; two reward modes {0,1,2}/{2,3,4}; temperature coefficient 2.5",
            "source_sha256": hashlib.sha256(
                Path(__file__).with_name("gflownet_nnx_tutorial.py").read_bytes()
            ).hexdigest(),
        }
    )
    with (output / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample", "action_1", "action_2", "action_3", "set"])
        for i, row in enumerate(actions):
            writer.writerow(
                [i + 1, *row.tolist(), "{" + ",".join(map(str, sorted(row))) + "}"]
            )
    # Keep exact source and environment metadata alongside the model for reproduction.
    for name in (
        "gflownet_nnx_tutorial.py",
        "utility.py",
        "experiment_report.py",
        "pyproject.toml",
        "uv.lock",
    ):
        (output / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report_text = (
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    (output / "metrics.json").write_text(report_text, encoding="utf-8")
    report_path = write_report(report, output)
    if args.publish_report is not None:
        args.publish_report.parent.mkdir(parents=True, exist_ok=True)
        args.publish_report.write_text(
            report_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        print(f"Published: {args.publish_report.resolve()}", flush=True)
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        with args.report_json.open("x", encoding="utf-8") as stream:
            stream.write(report_text)
    print(f"\nReport: {report_path.resolve()}", flush=True)
