# GFlowNet Tutorial

本项目用 uv 管理独立 `.venv`（Windows x86-64 CPU）。Python **3.13**；依赖见 `pyproject.toml` / `uv.lock`。

## 文件

- `gflownet_nnx_tutorial.py`：奖励、模型、采样、TB 损失与训练
- `utility.py`：CLI、诊断、checkpoint、实验产物
- `experiment_report.py`：SVG / HTML 报告渲染
- `gflownet_tutorial.html`：数学推导与静态教程
- `experiment_report.html`：最新训练实验的独立报告（根目录）
- `tests/`：数值 / 采样 / checkpoint / CPU 环境测试

## 安装与训练

```powershell
uv sync --locked
uv run python gflownet_nnx_tutorial.py --help
uv run python gflownet_nnx_tutorial.py --publish-report experiment_report.html
```

默认 CPU 训练 2,000 步（batch 256，width 64，Adam 0.002，ε=0.2，seed 0），保存并重载模型后用 seed 2026、ε=0 采 20,000 个集合。详细图表与样例见 `experiment_report.html`；完整运行产物在 `runs/<id>/`（已 gitignore）。

```powershell
uv run python gflownet_nnx_tutorial.py --checkpoint runs/<run>/model.msgpack
uv run python gflownet_nnx_tutorial.py --backward uniform
```

## 验证

```powershell
uv run ruff check gflownet_nnx_tutorial.py utility.py experiment_report.py tests
uv run ruff format --check gflownet_nnx_tutorial.py utility.py experiment_report.py tests
uv run pytest -q
```
