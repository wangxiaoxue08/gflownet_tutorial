"""Bounded integration checks for the installed CPU stack, not benchmarks."""

import math

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import optax
import optuna
import tensorflow as tf
from flax import nnx
from numpyro.infer import MCMC, NUTS


def test_cpu_devices() -> None:
    assert jax.devices()
    assert all(device.platform == "cpu" for device in jax.devices())
    assert tf.config.get_visible_devices("CPU")
    assert not tf.config.get_visible_devices("GPU")
    assert not tf.test.is_built_with_cuda()


def test_tf_data_nnx_training_step() -> None:
    """Move float32 [batch, feature] data to JAX outside the compiled step."""
    with tf.device("/CPU:0"):
        inputs = tf.constant([[1.0], [2.0], [3.0], [4.0]], dtype=tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices((inputs, 2.0 * inputs))
        host_inputs, host_targets = next(iter(dataset.batch(4).as_numpy_iterator()))
    x, y = jnp.asarray(host_inputs), jnp.asarray(host_targets)
    assert x.shape == y.shape == (4, 1)
    assert x.dtype == y.dtype == jnp.float32

    model = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
    model.train()
    optimizer = nnx.Optimizer(model, optax.sgd(0.01), wrt=nnx.Param)

    def loss_fn(model: nnx.Linear, x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.mean(jnp.square(model(x) - y))

    @nnx.jit
    def train_step(model, optimizer, x, y):
        loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
        optimizer.update(model, grads)
        return loss, grads

    before = jax.tree.map(lambda value: value.copy(), nnx.state(model, nnx.Param))
    loss, grads = train_step(model, optimizer, x, y)
    assert bool(jnp.isfinite(loss))
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(grads))
    after = nnx.state(model, nnx.Param)
    assert any(
        bool(jnp.any(old != new))
        for old, new in zip(
            jax.tree.leaves(before), jax.tree.leaves(after), strict=True
        )
    )
    model.eval()
    predictions = model(x)
    assert predictions.shape == y.shape
    assert bool(jnp.all(jnp.isfinite(predictions)))
    assert float(loss_fn(model, x, y)) < float(loss)


def test_numpyro_inference() -> None:
    """Check one short CPU chain; this does not establish convergence."""

    def model(observations: jax.Array) -> None:
        # Scalar Normal prior and likelihood; the plate indexes observations.
        location = numpyro.sample("location", dist.Normal(0.0, 1.0))
        with numpyro.plate("observations", observations.shape[0]):
            numpyro.sample("obs", dist.Normal(location, 1.0), obs=observations)

    sampler = MCMC(
        NUTS(model),
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        chain_method="sequential",
        progress_bar=False,
    )
    sampler.run(
        jax.random.key(1),
        jnp.array([0.1, 0.2, -0.1], dtype=jnp.float32),
        extra_fields=("potential_energy",),
    )
    samples = sampler.get_samples()["location"]
    assert samples.shape == (10,)
    assert bool(jnp.all(jnp.isfinite(samples)))
    assert bool(jnp.all(jnp.isfinite(sampler.get_extra_fields()["potential_energy"])))


def test_optuna_study() -> None:
    """Exercise study execution with a synthetic objective and two trials."""
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.RandomSampler(seed=2)
    )
    study.optimize(
        lambda trial: (trial.suggest_float("x", -1.0, 1.0) - 0.25) ** 2,
        n_trials=2,
        n_jobs=1,
    )
    assert len(study.trials) == 2
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    assert math.isfinite(study.best_value)
