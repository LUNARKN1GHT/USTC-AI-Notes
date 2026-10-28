# 实验六：卷积神经网络与 CIFAR-10 彩色图像分类

承接实验五，复用 PyTorch 的训练与评估流程，重点学习 RGB 通道、局部连接、权重共享，以及用对照实验判断模型和数据增强是否适合任务。

## 文件

| 文件 | 用途 |
| --- | --- |
| [实验六_学生版.ipynb](实验六_学生版.ipynb) | 学生练习：通道操作、卷积计算、CNN 搭建、模型比较和结果解释 |
| [实验六_参考答案.ipynb](实验六_参考答案.ipynb) | 完整代码、参考解释与实际运行输出 |
| [lab_utils.py](lab_utils.py) | 教师提供的数据划分、视图、训练、评估与绘图代码 |
| [实验六设计大纲.md](实验六设计大纲.md) | 教学安排、任务范围和对照条件 |
| [验证参考答案.py](验证参考答案.py) | 教师快速检查模型、指标汇总、参数恢复与数据隔离 |
| [参考运行记录.json](参考运行记录.json) | 本次 CPU quick 模式的实际配置、逐轮指标与测试结果 |

## 运行

在本目录启动 notebook，使用第五次已能导入 `torch`、`torchvision`、`numpy`、`matplotlib` 的 Python 内核。新建环境时，按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择适合机器的版本，再安装 NumPy、matplotlib 和 JupyterLab。运行 notebook 不需要安装 d2l。

第一次联网运行，将配置单元中的 `DOWNLOAD` 改为 `True`。数据缓存完整后改回 `False`，以后可以离线运行。也可以把教师准备的 `data/cifar-10-batches-py/` 文件夹复制到本目录。下载开关与 pip 软件源相互独立；不必反复下载。

默认 `MODE="quick"`：从固定的 45,000 张训练池中取 12,000 张，使用全部 5,000 张验证图像，各训练 8 轮。`MODE="full"` 使用全部训练池，各训练 12 轮。改变模式后应重新从头运行三组实验；不同模式的结果不能放在一起当作同预算比较。

默认 CPU，最多使用 4 个 CPU 线程。已有 CUDA 或 MPS 环境时可改 `DEVICE`，并记录设备。完整模式和其他设备的耗时需在对应机器上实测。

测试数据只在模型与轮次按验证集确定之后使用。参考答案保存 `outputs/best_cifar10.pt`，其中包含推理所需的模型参数、结构名称和预处理约定；它不包含优化器状态，不能用于精确续训。数据和输出目录已由本地 `.gitignore` 排除。

## 实验顺序

1. 拆分 RGB 通道，区分 HWC 与 BCHW，理解标准化与显示时的反标准化。
2. 核验一次局部卷积和最大池化，观察固定边缘滤波。
3. 构建 MLP 与两组卷积块的 CNN，核验形状和参数量。
4. 检查梯度、参数更新与固定小样本拟合，再完整训练两种模型。
5. 保持 CNN 初值、数据划分、批次顺序和训练预算一致，比较加入增强前后的验证表现。
6. 冻结选择，报告一次最终测试、混淆矩阵和错分样本。

数据增强采用反射填充后的随机裁剪与水平翻转。它只应用于训练视图；训练曲线、验证和测试均使用确定性预处理。CNN 与 MLP 同时改变多个结构因素，因此不能将差异单独归因于卷积；增强对照也不能把裁剪与翻转的作用分别拆开解释。

## 教师检查

在本目录运行：

```bash
python 验证参考答案.py
python 验证参考答案.py --check-data
```

第一条无需下载数据，检查模型、非整批指标汇总、无梯度评估、独立参数快照以及最佳验证轮次的恢复。第二条另检查本地数据划分、原图／增强视图的索引一致性、验证视图不受增强影响，以及绘图反标准化。

快速检查不能替代参考答案从头运行。参考 notebook 应保留完整执行输出；若修改模型或训练配置，需要重新生成参考结果。

本次实测使用 Python 3.9.6、PyTorch 2.8.0、torchvision 0.23.0，arm64 CPU、4 线程。quick 模式三组的最佳验证准确率为 MLP 46.12%、CNN 57.90%、CNN 加增强 55.84%，训练及逐轮评估分别约 9、29、36 秒。按验证选择无增强 CNN，最终测试准确率为 58.05%。这些是当前机器与配置下的参考观察，不是学生评分门槛，也不是 full 模式或其他设备的性能承诺。

## 参考资料

- [CIFAR-10 官方说明](https://cave.cs.toronto.edu/kriz/cifar.html)
- [d2l：从全连接层到卷积](https://zh.d2l.ai/chapter_convolutional-neural-networks/why-conv.html)
- [d2l：卷积神经网络（LeNet）](https://zh.d2l.ai/chapter_convolutional-neural-networks/lenet.html)
- [PyTorch：Training a Classifier](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
