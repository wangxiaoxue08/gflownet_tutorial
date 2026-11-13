"""Bounded CPU checks for the GFlowNet tutorial: training, sampling, checkpoints.

Imports follow the tutorial style: pull public names from
``gflownet_nnx_tutorial`` (which re-exports utility helpers) and
``experiment_report`` for HTML rendering.
"""

from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from itertools import permutations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from experiment_report import write_report
from gflownet_nnx_tutorial import (
    SetGFlowNet,
    evaluate,
    exact_diagnostics,
    generate_samples,
    load_model,
    parse_args,
    sample_actions,
    sample_backward,
    save_model,
    summarize_samples,
    tb_loss,
    train_step,
)


@pytest.fixture(scope="module")
def trained_model() -> SetGFlowNet:
    """Train a small-width model for 60 TB steps on all 60 paths.

    Returns:
        Eval-mode ``SetGFlowNet`` with finite loss decrease and updated params.
    """
    model = SetGFlowNet(rngs=nnx.Rngs(7), width=16)
    model.train()
    paths = jnp.asarray(list(permutations(range(5), 3)), dtype=jnp.int32)
    optimizer = nnx.Optimizer(model, optax.adam(0.01), wrt=nnx.Param)
    before = [np.asarray(x).copy() for x in jax.tree.leaves(nnx.state(model))]
    loss, grads = nnx.value_and_grad(tb_loss)(model, paths)
    assert bool(jnp.isfinite(loss))
    assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(grads))
    for _ in range(60):
        train_step(model, optimizer, paths)
    after_loss = float(tb_loss(model, paths))
    assert after_loss < float(loss) * 0.2
    assert any(
        not np.array_equal(old, np.asarray(new))
        for old, new in zip(before, jax.tree.leaves(nnx.state(model)), strict=True)
    )
    model.eval()
    return model


@pytest.mark.parametrize("learn_backward", [True, False])
def test_forward_backward_and_probability_mass(learn_backward: bool) -> None:
    """Forward/backward samples are legal; exact P_F and P_B masses normalize."""
    model = SetGFlowNet(rngs=nnx.Rngs(5), width=16, learn_backward=learn_backward)
    _, terminal = sample_actions(
        model, jax.random.key(1), jnp.zeros((32, 5)), jnp.array(0.0)
    )
    removals, source = sample_backward(model, jax.random.key(2), terminal)
    assert removals.shape == (32, 3)
    np.testing.assert_array_equal(source, np.zeros((32, 5)))
    assert np.all(np.diff(np.sort(np.asarray(removals), axis=1), axis=1) > 0)
    for child, row in zip(np.asarray(terminal), np.asarray(removals), strict=True):
        np.testing.assert_array_equal(child[row], np.ones(3))
    result = exact_diagnostics(model)
    assert len(result["target"]) == len(result["model"]) == 10
    np.testing.assert_allclose(sum(result["target"]), 1.0, atol=1e-6)
    np.testing.assert_allclose(sum(result["model"]), 1.0, atol=1e-6)
    np.testing.assert_allclose(result["backward_path_sums"], np.ones(10), atol=2e-6)
    # Independently evaluate the stated reward via symmetric-set distances.
    rewards = []
    for subset in result["subsets"]:
        rewards.append(
            sum(
                np.exp(-2.5 * len(set(subset).symmetric_difference(mode)) / 2)
                for mode in ({0, 1, 2}, {2, 3, 4})
            )
        )
    np.testing.assert_allclose(result["target"], rewards / np.sum(rewards), rtol=1e-6)


def test_saved_model_generates_same_samples(
    trained_model: SetGFlowNet, tmp_path
) -> None:
    """Checkpoint round-trip keeps exact probs and same-key samples."""
    path = tmp_path / "model.msgpack"
    save_model(trained_model, path, {"seed": 7, "steps": 60}, [])
    restored, metadata = load_model(path)
    assert metadata["training_config"]["seed"] == 7
    np.testing.assert_array_equal(
        exact_diagnostics(trained_model)["model"],
        exact_diagnostics(restored)["model"],
    )
    expected = generate_samples(trained_model, 137, 44, batch_size=64)
    actual = generate_samples(restored, 137, 44, batch_size=64)
    np.testing.assert_array_equal(expected, actual)
    assert actual.shape == (137, 3)  # Incomplete final batch must be truncated.
    different = generate_samples(restored, 137, 45, batch_size=64)
    assert not np.array_equal(actual, different)
    with pytest.raises(FileExistsError):
        save_model(trained_model, path, {}, [])


def test_sample_examples_and_distribution(trained_model: SetGFlowNet, tmp_path) -> None:
    """Empirical counts match samples; report HTML/SVG/CSV stay consistent."""
    actions = generate_samples(trained_model, 4096, 2026, batch_size=512)
    args = parse_args(["--samples", "4096", "--width", "16"])
    report = evaluate(trained_model, args, [], actions)
    assert sum(report["counts"]) == 4096
    assert report["counts"] == [
        sum(sorted(row.tolist()) == subset for row in actions)
        for subset in report["subsets"]
    ]
    np.testing.assert_allclose(report["empirical"], np.asarray(report["counts"]) / 4096)
    np.testing.assert_array_equal(report["examples"][0]["actions"], actions[0])
    assert report["tv_sampling"] < 0.08  # Generous bound for ten bins at N=4096.
    assert np.all(np.asarray(report["interval_low"]) <= report["empirical"])
    assert np.all(np.asarray(report["interval_high"]) >= report["empirical"])
    report.update(
        {
            "run_id": "test-run",
            "created_utc": "test",
            "training_seconds_including_compile": None,
            "restore_max_probability_error": 0.0,
            "training_config": {
                "steps": 60,
                "batch_size": 60,
                "seed": 7,
                "width": 16,
                "learning_rate": 0.01,
                "epsilon": 0.0,
                "backward": "learned",
                "log_every": 60,
            },
            "sampling_config": {"n": 4096, "seed": 2026, "batch_size": 512},
            "hardware": {"backend": "cpu"},
            "task": "test",
            "source_sha256": "test",
            "checkpoint_sha256": "test",
        }
    )
    html_path = write_report(report, tmp_path)
    html = html_path.read_text(encoding="utf-8")
    assert "4,096" in html and "<svg" in html and "samples.csv" in html
    for filename in ("distribution.svg", "training.svg"):
        root = ET.parse(tmp_path / filename).getroot()
        assert root.tag == "{http://www.w3.org/2000/svg}svg"
    with (tmp_path / "distribution.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    assert sum(int(row["count"]) for row in rows) == 4096
    np.testing.assert_allclose(
        [float(row["sample_frequency"]) for row in rows], report["empirical"]
    )


def test_sample_validation() -> None:
    """summarize_samples rejects repeated items and empty batches."""
    with pytest.raises(ValueError, match="distinct"):
        summarize_samples(np.array([[0, 0, 1]]))
    with pytest.raises(ValueError, match="nonempty"):
        summarize_samples(np.empty((0, 3), dtype=np.int32))


def test_cli_rejects_invalid_training_settings() -> None:
    """parse_args exits on non-finite LR, out-of-range epsilon, or non-positive width."""
    for argv in (["--learning-rate", "nan"], ["--epsilon", "1.1"], ["--width", "0"]):
        with pytest.raises(SystemExit):
            parse_args(argv)
