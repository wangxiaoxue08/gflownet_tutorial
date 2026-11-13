# Link tutorial and report source references to GitHub

## Current task
- Goal: point code-file references in both HTML pages to the repository's
  `master` branch; preserve executable commands and experiment artifact links.
- Update the report renderer too so regenerated reports retain source links.
- Steps: [done] update both HTML pages and report renderer; [done] verify changes.
- Verification: all three source targets are present in both HTML pages and the
  renderer, with no remaining local source links; diff check and Ruff lint/format
  checks passed; `uv run pytest -q tests/test_gflownet.py`: 6 passed on CPU.

## Previous task: restyle tests, regenerate root report, cleanup

## Goal and constraints
- Restyle `tests/test_gflownet.py` like the tutorial (`from ... import`).
- Retrain with default settings; publish standalone HTML to repo-root
  `experiment_report.html`.
- Align tutorial `#run` facts with the new report; drop stale embedded source.
- Remove temporary sync scripts and keep docs lean. Numerical training unchanged.

## Steps and status
1. [done] Restyle tests; add `--publish-report`.
2. [done] Train learned + uniform; publish root report; sync tutorial tables/commands.
3. [done] Slim PLAN/README; delete temp helpers; run lint/tests.

## Verification / outcome
- Pre-commit verification: Ruff lint and format checks passed for all six Python
  files; `uv run pytest -q` passed all 10 tests on CPU.
- Learned run: `runs/gflownet-20261113-seed0/` → root `experiment_report.html`
  (TV_exact 0.0000191, TV_empirical 0.0091783, log Z 1.1069663).
- Uniform run: `runs/gflownet-20261113-uniform/` (TV_exact 0.0000036,
  TV_empirical 0.0055630).
- Tutorial `#run` now matches these metrics and project `uv run python` workflow;
  removed outdated PEP 723 / JAX 0.7.2 claims and the stale base64 full-source dump.
- Remaining tutorial code excerpts are pedagogical snippets (may omit CLI /
  utility split); canonical code is the `.py` files.
