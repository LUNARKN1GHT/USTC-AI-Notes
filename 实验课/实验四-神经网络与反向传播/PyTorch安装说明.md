# 实验四：PyTorch 安装说明

本说明用于 Q10 的自动求导演示。**Q10 不算分，CPU 就能运行；不安装 PyTorch 也能完成 Q1–Q9。** 本实验只需要 `torch`，无需安装用于图像和音频任务的 `torchvision`、`torchaudio`。

PyTorch 仍然可以用 pip 安装，包名是 `torch`。与安装 NumPy、Pandas 相比，需要多确认几件事：**Python 版本与系统架构是否受支持、需要 CPU 还是 GPU、软件源是否提供对应的安装包**。镜像解决下载问题，不能解决版本或硬件不兼容的问题。

## 1. 先确认装到哪个环境

在终端中激活此前实验使用的 Conda 环境；Windows 可以使用 Anaconda Prompt：

```bash
conda activate ai-lab
python --version
python -c "import sys, platform, struct; print('Python:', sys.executable); print('系统:', platform.system(), platform.machine()); print('位数:', struct.calcsize('P') * 8)"
python -m pip --version
```

确认 Python 来自 `ai-lab`，并使用受支持的 64 位 Python。若需要新建环境，建议选择 Python 3.11；已有环境能安装兼容版本时，可以继续使用。不同 PyTorch 版本支持的 Python 范围不同，不要把“最新版本的要求”当成所有历史版本的要求。

先通过科大镜像更新当前环境中的 pip：

```bash
python -m pip install --upgrade pip -i https://mirrors.ustc.edu.cn/pypi/simple
```

这里使用 `python -m pip`，让安装工具对应当前的 Python，减少“装好了，但 Notebook 找不到”的情况。以下安装命令在终端运行，**按自己的情况选一条路线即可**。

## 2. 选择安装路线

| 电脑／需求 | 建议路线 |
| --- | --- |
| Windows／Linux，完成本实验 | CPU 专用版，见 2.2 |
| Apple 芯片 Mac（M 系列） | 科大 PyPI 镜像，见 2.1；Q10 使用 CPU 即可 |
| Intel Mac | 兼容的旧版本，见 2.3 |
| Windows／Linux，有 NVIDIA 显卡且希望使用 GPU | CUDA 版，见 2.4；本实验无需 GPU |

### 2.1 通过科大 PyPI 镜像安装

适用于 Apple 芯片 Mac，也可用于受支持的 Windows／Linux 环境：

```bash
python -m pip install torch -i https://mirrors.ustc.edu.cn/pypi/simple
```

`-i` 指定本次安装使用科大 PyPI 镜像，不必修改全局配置。pip 会选择与当前 Python、操作系统和架构匹配的包。

**这条命令不保证安装 CPU 专用版。** 尤其在 Linux 上，PyPI 中的默认包可能附带较大的 CUDA 依赖；即使没有 NVIDIA 显卡，也能用它进行 CPU 计算，但下载和磁盘占用可能更大。如果希望明确安装 CPU 专用版，使用下一节。

### 2.2 Windows／Linux：明确安装 CPU 专用版

CPU 专用包使用 PyTorch 官方的 CPU 索引。普通科大 PyPI 镜像不能直接替代这个专用索引：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

上一步更新 pip，以及实验所需的 NumPy、Matplotlib 等普通依赖，仍可走科大镜像。这里的 `--index-url` 与 `-i` 是同一个选项；不要把 CPU 索引替换为科大 PyPI 地址后，仍认为自己明确选择了 CPU 专用版。

若此前已装过其他构建，想切换到 CPU 专用版，先在当前环境执行 `python -m pip uninstall torch`，再运行上述命令。

### 2.3 Intel Mac：使用兼容版本

PyTorch 官方预编译包对 Intel Mac 的支持止于 2.2.x。建议在 Python 3.11 环境中，通过科大镜像安装：

```bash
python -m pip install "numpy<2" "torch==2.2.2" -i https://mirrors.ustc.edu.cn/pypi/simple
```

限制 NumPy 为 1.x，是为了避免旧版 PyTorch 与 NumPy 2 的二进制兼容问题，保证 Q10 中 `.numpy()` 转换可用。Apple 芯片 Mac 应使用原生 ARM64 Python；如果检查结果为 `x86_64`，先确认是否误用了 Intel 版 Python 或 Rosetta 环境。

### 2.4 选读：NVIDIA GPU 与 CUDA 版

CUDA 路线用于受支持的 Windows／Linux NVIDIA 显卡。Apple GPU 和普通 Intel／AMD 核显不能套用这条路线；Apple GPU 使用的是另一套 MPS 后端。

先在终端执行：

```bash
nvidia-smi
```

确认系统能识别显卡及驱动，再打开 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)，选择 Stable、自己的操作系统、Pip、Python 和适配驱动与显卡的 CUDA 构建。复制生成的命令，将开头的 `pip` 或 `pip3` 改为 `python -m pip`；本实验只保留 `torch` 即可。

注意：

- `nvidia-smi` 中的 “CUDA Version” 表示驱动所支持的 CUDA 能力，并不表示本机已安装同版本 CUDA Toolkit，也不要求 PyTorch 的 CUDA 构建版本与它逐字一致。
- 使用官方预编译包运行本实验，通常无需另装 CUDA Toolkit 或 cuDNN；需要的是兼容的 NVIDIA 驱动。自行编译 PyTorch 或 CUDA 扩展才有额外要求。
- 科大 PyPI 镜像与官方 CUDA 专用索引是不同的软件源。不要只替换 URL、随意填写 `cuXXX`，或以为加上科大 `-i` 就指定了 CUDA 版本。
- 安装 GPU 版不会自动让所有计算转到 GPU。本实验 Q10 的代码默认在 CPU 上运行，无需为了安装 GPU 版而改写实验。

## 3. 在 Notebook 中验证

安装完成后，选择对应的 `ai-lab` 内核并**重启内核**，在新单元格中执行：

```python
import sys
import torch

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("安装位置:", torch.__file__)
print("CUDA 构建版本:", torch.version.cuda)
print("CUDA 可用:", torch.cuda.is_available())

# Q10 需要张量计算、自动求导，以及与 NumPy 的转换。
x = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
loss = (x ** 2).sum()
loss.backward()
print("梯度:", x.grad)                     # 应为 tensor([4.], dtype=torch.float64)
print("转换为 NumPy:", x.detach().numpy())  # 应为 [2.]
```

能导入 `torch`、梯度为 4、能转为 NumPy，即通过本实验所需的基本环境检查。CPU 版或 Mac 上 `CUDA 可用` 为 `False` 属于正常现象；只有选择 NVIDIA GPU 路线时，才需要进一步检查它为何不是 `True`。

## 4. 遇到错误时，先定位再处理

| 现象 | 优先检查与处理 |
| --- | --- |
| `No matching distribution found` | 核对 Python 版本、系统架构和选定构建；更新 pip。若固定版本不支持当前 Python，使用兼容环境或参考官方历史版本说明，不要反复换源。 |
| 终端能导入，Notebook 报 `No module named 'torch'` | 比较两处 `sys.executable`，选择正确内核并重启。也可在目标 Notebook 中用 `%pip install torch -i https://mirrors.ustc.edu.cn/pypi/simple` 安装默认构建，再重启内核。 |
| 下载超时 | 科大源命令可加 `--timeout 120` 后重试；下载 CUDA 包时同时关注磁盘空间。CPU／CUDA 专用索引的网络问题，应保留构建选择再寻找对应镜像。 |
| CUDA 版已安装，但 `torch.cuda.is_available()` 为 `False` | 检查是否有受支持的 NVIDIA 显卡、`nvidia-smi` 是否正常、驱动是否兼容，以及实际安装的构建。Q10 可以继续用 CPU。 |
| Intel Mac 出现 NumPy 初始化／转换错误 | 按 2.3 安装 `numpy<2`，重启内核后重新验证。 |

可以使用 AI 辅助排查。提供**操作系统、Python 版本与路径、显卡型号（若涉及 GPU）、完整安装命令和报错文本**，让它先判断是环境、版本、驱动还是下载问题。不要只说“装不上”，也不要在未理解原因时连续执行多组互相冲突的安装命令。

## 参考资料

- [科大 PyPI 镜像使用说明](https://mirrors.ustc.edu.cn/help/pypi.html)：镜像地址与 pip 配置。
- [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)：当前支持的平台与安装命令。
- [PyTorch 历史版本安装说明](https://pytorch.org/get-started/previous-versions/)：指定版本与计算平台的组合。
- [PyTorch 2.2 发布说明](https://pytorch.org/blog/pytorch2-2/)：Intel Mac 预编译包支持范围。

安装源和平台支持可能变化，以以上官方说明为准。
