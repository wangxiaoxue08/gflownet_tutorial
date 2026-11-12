# CPU environment setup

## Goal and constraints
- Install the AGENTS.md stack with uv on this Windows x86-64 CPU-only PC.
- Use JAX, Flax NNX, Optax, NumPyro, tensorflow-cpu (tf.data), and Optuna;
  add Ruff and pytest as required development tools. Do not install GPU extras.
- Preserve existing files and keep changes limited to environment setup and verification.
- Replace the initial Python 3.14 requirement with Python 3.13: TensorFlow CPU
  2.21.0 publishes Windows wheels through CPython 3.13, not 3.14.

## Steps and status
1. [done] Inspect repository instructions, configuration, available Python, and CPU wheels.
2. [done] Configure Python 3.13, install runtime/development dependencies, and create uv.lock/.venv.
3. [done] Add setup instructions and bounded CPU smoke tests for the required stack.
4. [done] Run formatting, lint, tests, locked sync, and review final changes.

## Verification
- `uv sync --locked`: passed; 95 installed packages checked.
- `uv run ruff check main.py tests`: passed.
- `uv run ruff format --check main.py tests`: passed (3 files).
- `uv run pytest -q tests/test_cpu_stack.py`: 4 passed in 47.80s.
  Covers CPU devices, tf.data batches, NNX forward/loss/gradients/optimizer update,
  small NumPyro inference, and a two-trial Optuna study.
- Verified on Windows x86-64 with Python 3.13.5, JAX/jaxlib 0.12.0.1,
  Flax 0.12.12, Optax 0.2.9, NumPyro 0.22.0, TensorFlow CPU 2.21.0,
  and Optuna 5.0.0. Exact dependencies are recorded in uv.lock.
- Reviewed setup files; existing tutorial Python/HTML files remain unchanged.
- No full training, convergence claims, GPU tests, or performance benchmarks.
