# GFlowNet Tutorial

本项目使用 uv 管理独立的 `.venv`，面向 Windows x86-64、无 GPU 的 CPU 环境。

## 技术栈

- JAX / jaxlib：CPU 数值计算、JIT 与自动微分。
- Flax `nnx` + Optax：模型与优化器。
- NumPyro：概率建模与推断。
- `tensorflow-cpu`：仅用 `tf.data` 构建数据管线。
- Optuna：超参数搜索。
- Ruff / pytest：开发检查与环境冒烟测试。

直接依赖在 `pyproject.toml` 中声明，完整版本由 `uv.lock` 固定。
JAX 使用默认 CPU 安装，不启用 CUDA/TPU extras；TensorFlow 使用明确的 CPU 包。
TensorFlow 自带的 Keras 是传递依赖，项目模型仍使用 Flax NNX。

Python 固定为 **3.13**（`>=3.13,<3.14`）。初始化配置中的 3.14 已调整：
TensorFlow CPU 2.21.0 的 Windows wheel 支持到 CPython 3.13。
参见 [TensorFlow CPU 发布文件](https://pypi.org/project/tensorflow-cpu/2.21.0/#files)
与 [JAX CPU 安装说明](https://docs.jax.dev/en/latest/installation.html#cpu)。

## 安装与运行

在项目目录使用 PowerShell：

```powershell
uv sync --locked
uv run python main.py
```

`uv sync` 会按 `.python-version` 选择 Python，并在需要时下载解释器、创建 `.venv`。
默认包含开发依赖。不需要手动激活虚拟环境，也不需要安装 CUDA/cuDNN。
`main.py` 目前是初始化占位程序；用下面的测试验证完整环境。

## 验证

```powershell
uv run ruff check main.py tests
uv run ruff format --check main.py tests
uv run pytest -q
```

测试会在数值库初始化前选择 CPU，覆盖设备检测、`tf.data` 到 JAX 的批次转换、
NNX 前向计算/损失/梯度/Optax 参数更新、短 NumPyro NUTS 链，以及两次 Optuna trial。
这些是小规模环境检查，不代表推断收敛或性能评估。

后续脚本可在导入数值库之前显式设置 CPU：

```python
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")  # 必须早于 TensorFlow 设备初始化
```

数据迭代与 `.as_numpy_iterator()` 放在 JIT 外，在输入边界转换成 JAX 数组。
