# TimesFM

TimesFM（时间序列基础模型）是Google Research为时间序列预测开发的预训练时间序列基础模式。

* 论文: [时间序列预测的纯解码器基础模型](https://arxiv.org/abs/2310.10688), 将出现在ICML 2024.
* [谷歌研究博客](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
* [Hugging Face 检查点仓库](https://huggingface.co/google/timesfm-1.0-200m)

此仓库包含加载公共TimesFM检查点和运行模型的代码
推论请访问我们的下载模型检查点。
[Hugging Face checkpoint repo](https://huggingface.co/google/timesfm-1.0-200m)
这不是Google官方支持的产品。
我们建议至少16GB RAM来加载TimesFM依赖项。

## Checkpoint timesfm-1.0-200m

timesfm-1.0-200m 是第一个开放模型检查点:

- 它对多达512个时间点的上下文长度和任何范围长度进行单变量时间序列预测，并带有可选的示频器。
- 它侧重于点预测，不支持概率预测。我们实验性地提供了分位数头，但它们在预训练后尚未校准。
- 它要求上下文是连续的（即没有“洞”），上下文和范围的频率相同。

## Benchmarks

请参阅我们的结果表 [扩展基准测试](https://github.com/google-research/timesfm/tree/master/experiments/extended_benchmarks) and the [long horizon benchmarks](https://github.com/google-research/timesfm/tree/master/experiments/long_horizon_benchmarks).

请查看“experiments/”中相应基准目录中的README文件，了解在相应基准上运行TimesFM的说明。

## 安装说明

### 打包安装

要将TimesFM作为软件包安装，您可以在不克隆此仓库的情况下运行以下命令：

`pip install timesfm`

### 使用conda安装

为了调用TimesFM，我们有两个环境文件。在“timesfm”内部，for GPU安装（假设CUDA 12已安装），您可以创建conda
通过以下方式从基本文件夹中删除环境“tfm_env”：

```
conda env create --file=environment.yml
```

对于CPU设置，请使用，

```
conda env create --file=environment_cpu.yml
```
创造一个开发环境，然后执行
```
conda activate tfm_env
pip install -e .
```
以安装该软件包。

**注意**: 

1. 运行提供的基准测试需要额外的依赖关系。
请使用`experiments`下的环境文件。 

2. 依赖项“lingvo”不支持ARM架构，并且该代码不适用于使用苹果芯片的机器。我们意识到这个问题，并正在努力寻求解决方案。敬请期待。

### poetry的本地安装方法

从当前存储库/本地版本（就像您之前使用`pip-e.`一样），您可以运行以下命令

```
pip install poetry # optional
poetry install
```

这将在本地安装环境.venv 文件夹 (取决于实际配置) 并于poetry环境的python命令相匹配. 如果不这样的话，您可以使用“poetry run python”来使用本地环境。

### 注意

1. 运行提供的基准测试需要额外的依赖关系。请使用`experiments` 下的环境文件。

2. 依赖项“lingvo”不支持ARM架构，并且该代码不适用于使用苹果芯片的机器。我们意识到这个问题，并正在努力寻求解决方案。敬请期待。

#### 构建包并发布到PyPI

可以使用命令“poetry build”构建该包。

要构建并发布到PyPI，可以使用命令“poetry publish”。此命令将要求用户具有发布到PyPI存储库所需的权限。

## 用法

### 构建初始化模型并加载检查点。
然后，基类可以被加载为，

```python
import timesfm

tfm = timesfm.TimesFm(
    context_len=<context>,
    horizon_len=<horizon>,
    input_patch_len=32,
    output_patch_len=128,
    num_layers=20,
    model_dims=1280,
    backend=<backend>,
)
tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
```
请注意，这四个参数是固定的，用于加载200m的模型
```python
input_patch_len=32,
output_patch_len=128,
num_layers=20,
model_dims=1280,
```

1. 这里的`context_len`可以设置为模型**的最大上下文长度**。**它需要是“input_patch_len”的倍数，即32的倍数。**您可以向“tfm.predictor（）”函数提供一个较短的序列，模型将处理它。目前，模型处理的最大上下文长度为512，在以后的版本中会增大。输入时间序列可以具有**任何上下文长度**。如果需要，填充/截断将由推理代码处理。

2. 范围长度可以设置为任何值。我们建议将其设置为应用程序预测任务中所需的最大水平长度。我们通常建议范围长度<=上下文长度，但这不是函数调用中的要求。

3. `backend`是“cpu”、“gpu”或“tpu”之一，注意区分大小写。

### 执行推理

我们提供API，从数组输入或“pandas”数据帧进行预测。这两种预测方法都期望（1）输入时间序列上下文，（2）以及它们的频率。请查看函数“tfm.predictor（）”和“tfm.forecast_on_df（）”的文档以获取详细说明。

特别是关于频率，TimesFM期望一个值为｛0，1，2｝的分类指标：

- **0** （默认）：高频、长时间序列。我们建议将其用于高达每日粒度的时间序列。
- **1**: 中频时间序列。我们建议将其用于每周和每月的数据。
- **2**: 低频、短时间序列。我们建议将其用于每月以外的任何事情，例如每季度或每年。

这个分类值应该直接与数组输入一起提供。对于数据帧输入，我们将频率的传统字母编码转换为预期的类别，即

- **0**: T, MIN, H, D, B, U
- **1**: W, M
- **2**: Q, Y

请注意，您**不**必须严格遵守我们的建议。虽然这是我们在模型训练期间的设置，我们希望它能提供最佳的预测结果，但您也可以将频率输入视为自由参数，并根据您的特定用例进行修改。


示例：
阵列输入，频率分别设置为低、中、高。

```python
import numpy as np
forecast_input = [
    np.sin(np.linspace(0, 20, 100)),
    np.sin(np.linspace(0, 20, 200)),
    np.sin(np.linspace(0, 20, 400)),
]
frequency_input = [0, 1, 2]

point_forecast, experimental_quantile_forecast = tfm.forecast(
    forecast_input,
    freq=frequency_input,
)
```

`pandas`的数据帧，频率设置为每月“M”。

```python
import pandas as pd

# e.g. input_df is
#       unique_id  ds          y
# 0     T1         1975-12-31  697458.0
# 1     T1         1976-01-31  1187650.0
# 2     T1         1976-02-29  1069690.0
# 3     T1         1976-03-31  1078430.0
# 4     T1         1976-04-30  1059910.0
# ...   ...        ...         ...
# 8175  T99        1986-01-31  602.0
# 8176  T99        1986-02-28  684.0
# 8177  T99        1986-03-31  818.0
# 8178  T99        1986-04-30  836.0
# 8179  T99        1986-05-31  878.0

forecast_df = tfm.forecast_on_df(
    inputs=input_df,
    freq="M",  # monthly
    value_name="y",
    num_jobs=-1,
)
```

## 微调

我们在中提供了一个在新数据集上微调模型的示例`notebooks/finetuning.ipynb`.

## 贡献风格指南

如果您想提交PR，请确保使用我们的格式样式。我们使用[yapf](https://github.com/google/yapf)对于使用以下选项进行格式化，

```
[style]
based_on_style = google
# Add your custom style rules here
indent_width = 2
spaces_before_comment = 2

```

请运行`yapf --in-place --recursive <filename>`命令在所有受影响的文件.
