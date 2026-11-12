# AGENTS.md

## Workflow and planning

- Read the relevant repository instructions, README, configuration, and nearby code before editing. Follow applicable nested `AGENTS.md` files.
- Always create or update `PLAN.md` before making code changes for a non-trivial task. This includes new features, behavioral changes, dependency changes, and work spanning multiple components. Trivial typo or formatting fixes do not require a plan.
- Keep `PLAN.md` concise: goal, constraints, implementation steps, current status, and verification. Update it at meaningful milestones and when the task is complete; do not turn it into a transcript.
- Make the smallest complete change that satisfies the request. Preserve unrelated user changes and existing public interfaces unless the task requires changing them.
- Proceed through implementation and verification without pausing for routine implementation choices. Ask only when missing information materially affects correctness or scope.

## Required tools and environment

| Purpose | Required tool |
| --- | --- |
| Deep learning | JAX, `flax.nnx`, and Optax |
| Probabilistic modeling and inference | NumPyro |
| Automatic differentiation | JAX, including JAX-backed NNX transformations |
| Dataset loading and input pipelines | `tensorflow.data` (`tf.data`) |
| Hyperparameter search | Optuna |
| Python environments and dependencies | `uv` |
| Running Python scripts and project tools | `uv run` |

- Use `uv add`, `uv remove`, and `uv sync` for dependency management. Keep `pyproject.toml` and `uv.lock` consistent; respect the project's Python version.
- Run scripts with `uv run python ...` and tools with `uv run <tool> ...`. Do not install into the system Python environment.
- Use `rg` and `rg --files` for searching code and files.
- Follow existing lint, formatting, testing, and type-checking configuration. For new Python projects without established tooling, use Ruff and pytest.
- Check installed or locked package versions before using APIs. Consult official documentation for those versions when uncertain, especially for Flax NNX. Do not upgrade dependencies just to match an example.
- Add only dependencies needed for the task; the table does not require every project to install the entire stack.

## Coding conventions

- Follow the existing project structure. Use descriptive names, small functions, type hints for public interfaces, and docstrings for non-obvious behavior.
- Document array shapes, axis meanings, and dtype assumptions at model and numerical interfaces. Make units explicit where relevant.
- Keep data loading, model definitions, training or inference, and evaluation separable. Avoid introducing abstractions that the task does not need.
- Make seeds, paths, hyperparameters, and compute settings configurable. Reuse the project's configuration approach rather than adding a new framework.
- Use `jax.numpy` for differentiable numerical computation. NumPy is acceptable for host-side preprocessing, inspection, and serialization outside transformed computations.
- Manage randomness explicitly with JAX keys or NNX RNG streams. Derive independent keys for independent draws and preserve advancing RNG state across steps.
- Keep trainable parameters, non-trainable model state, optimizer state, and RNG state explicit. Use transformations compatible with the installed NNX version and the chosen state representation.
- Keep file I/O, dataset iteration, and ordinary logging outside JIT-compiled numerical kernels. Avoid implicit host/device transfers in hot paths.
- Set training and evaluation modes correctly for stateful or stochastic layers. Evaluation must not unintentionally update training state.
- When implementing resumable checkpoints, save the model state, optimizer state, step, RNG state, and configuration needed to resume. Preserve data-iteration state when exact continuation is required.

## Data and experiment conventions

- Use `tf.data` for dataset pipelines. Run TensorFlow data loading on CPU by default and disable TensorFlow GPU visibility before its GPU runtime initializes, leaving accelerators available to JAX.
- Convert batches at a clear input boundary outside JIT. Any preprocessing that must participate in autodifferentiation belongs in JAX.
- Keep train, validation, and test splits separate. Fit preprocessing on training data only; use group-aware or chronological splits when the task requires them.
- Use validation metrics for Optuna objectives and model selection. Reserve test data for final evaluation.
- Make NumPyro priors, likelihoods, parameter constraints, and batch/event dimensions explicit. Check `plate` structure and minibatch likelihood scaling when applicable.
- Record the seed, configuration, dataset version or split identifiers, package versions, and relevant hardware with substantive experiment results. Store outputs under distinct run identifiers.
- Keep verification runs small and bounded. Run full training, long inference, or broad hyperparameter sweeps only within the requested scope and compute budget.

## Verification

- Start with the smallest check that exercises the changed behavior. Expand verification only for a concrete remaining risk or a repository-required gate.
- Run the configured formatter, linter, relevant tests, and existing type checks as applicable. With Ruff and pytest, use these command forms, replacing placeholders with actual paths:

  ```bash
  uv run ruff check <changed_paths>
  uv run ruff format --check <changed_paths>
  uv run pytest -q <relevant_test_paths>
  ```

- Add focused regression tests for behavioral bugs and meaningful tests for new logic. Test observable behavior and numerical invariants; avoid tests that merely duplicate the implementation. Documentation-only changes do not require model tests.
- For training changes, run a small forward pass, loss computation, gradient computation, and optimizer step. Check shapes, finite outputs and gradients, and expected parameter updates. Use a small learning check when training behavior changes.
- For inference changes, run a small NumPyro smoke test and check finite values and sample shapes. A smoke test does not establish convergence: substantive MCMC results require diagnostics such as divergences, effective sample sizes, and R-hat across suitable chains. Assess SVI optimization and posterior predictive behavior when applicable.
- When checkpointing changes, verify save, restore, and continuation. When sharding changes, verify placement and execution on the intended device setup when available.
- For performance claims, separate compilation from steady-state execution, warm up the computation, and synchronize asynchronous JAX results with `block_until_ready()` before recording elapsed time.
- If a check cannot run because dependencies, data, or hardware are unavailable, state exactly what was skipped and why. Do not claim GPU, multi-device, convergence, or performance validation from a CPU smoke test.
- Before finishing, review the diff for unintended changes, update `PLAN.md` if one was required, and report the changes, checks run, outcomes, and remaining limitations.

## What to avoid

- Do not substitute PyTorch, Keras, Flax Linen, TensorFlow Probability, or another framework for the required stack unless the user explicitly requests it. Do not rewrite unrelated legacy code merely to enforce the stack.
- Do not introduce pip, Conda, or Poetry as competing project environment managers.
- Do not use NumPy or TensorFlow operations on JAX tracers, reuse random keys for independent draws, or mutate unsupported state inside transformations.
- Do not silently change precision, device allocation, model semantics, or loss normalization. Make necessary changes explicit and verify their effect.
- Do not hide NaNs, inference failures, or exceptions with blanket suppression or arbitrary clipping. Diagnose the cause.
- Do not fabricate metrics, test results, benchmark numbers, or completed experiments.
- Avoid unrelated refactors, repository-wide formatting, broad dependency upgrades, and speculative infrastructure.
- Do not overwrite unrelated work or existing experiment artifacts. Keep secrets, private datasets, large checkpoints, and generated outputs out of version control unless explicitly required and appropriately configured.
