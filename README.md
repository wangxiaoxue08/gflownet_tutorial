# GFlowNet Tutorial

这是一个用 Trajectory Balance 训练 GFlowNet 的小例子：从 5 个元素里无序选出 3 个。数学推导在 `gflownet_tutorial.html`，可跑的代码在 Python 文件里。

环境由 uv 管理，面向没有 GPU 的 Windows CPU 机器，Python 版本是 3.13。依赖写在 `pyproject.toml` 里，具体版本锁在 `uv.lock`。

## 仓库里有什么

- `gflownet_nnx_tutorial.py`：奖励函数、网络、采样、TB 损失和训练循环
- `utility.py`：命令行参数、评估、模型读写，以及把实验结果写到磁盘
- `experiment_report.py`：把结果画成 SVG，并生成可离线打开的 HTML 报告
- `gflownet_tutorial.html`：教程正文（推导 + 代码讲解）
- `experiment_report.html`：最近一次完整实验的报告，和上面几个文件放在同一层目录
- `tests/`：数值、采样、模型保存，以及 CPU 环境是否装对的检查

## 怎么跑

先装好依赖，再训练：

```powershell
uv sync --locked
uv run python gflownet_nnx_tutorial.py --help
uv run python gflownet_nnx_tutorial.py --publish-report experiment_report.html
```

默认会在 CPU 上训练 2000 步（batch 256，隐藏层宽度 64，Adam 学习率 0.002，探索率 0.2，seed 0）。训完后会把模型存下来再加载一遍，然后用 seed 2026、关闭探索，采 20000 个集合。图表和样例在 `experiment_report.html` 里；更完整的中间文件会写到 `runs/<某次运行>/`（这个目录已被 gitignore）。

如果只想从已有模型采样，或者试固定均匀反向策略：

```powershell
uv run python gflownet_nnx_tutorial.py --checkpoint runs/<run>/model.msgpack
uv run python gflownet_nnx_tutorial.py --backward uniform
```

## 检查一下环境有没有问题

```powershell
uv run ruff check gflownet_nnx_tutorial.py utility.py experiment_report.py tests
uv run ruff format --check gflownet_nnx_tutorial.py utility.py experiment_report.py tests
uv run pytest -q
```
