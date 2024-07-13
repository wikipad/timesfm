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

1. The `context_len` here can be set as the max context length **of the model**. **It needs to be a multiplier of `input_patch_len`, i.e. a multiplier of 32.** You can provide a shorter series to the `tfm.forecast()` function and the model will handle it. Currently, the model handles a max context length of 512, which can be increased in later releases. The input time series can have **any context length**. Padding / truncation will be handled by the inference code if needed.

2. The horizon length can be set to anything. We recommend setting it to the largest horizon length you would need in the forecasting tasks for your application. We generally recommend horizon length <= context length but it is not a requirement in the function call.

3. `backend` is one of "cpu", "gpu" or "tpu", case sensitive.

### Perform inference

We provide APIs to forecast from either array inputs or `pandas` dataframe. Both forecast methods expect (1) the input time series contexts, (2) along with their frequencies. Please look at the documentation of the functions `tfm.forecast()` and `tfm.forecast_on_df()` for detailed instructions.

In particular regarding the frequency, TimesFM expects a categorical indicator valued in {0, 1, 2}:

- **0** (default): high frequency, long horizon time series. We recommend using this for time series up to daily granularity.
- **1**: medium frequency time series. We recommend using this for weekly and monthly data.
- **2**: low frequency, short horizon time series. We recommend using this for anything beyond monthly, e.g. quarterly or yearly.

This categorical value should be directly provided with the array inputs. For dataframe inputs, we convert the conventional letter coding of frequencies to our expected categories, that

- **0**: T, MIN, H, D, B, U
- **1**: W, M
- **2**: Q, Y

Notice you do **NOT** have to strictly follow our recommendation here. Although this is our setup during model training and we expect it to offer the best forecast result, you can also view the frequency input as a free parameter and modify it per your specific use case.


Examples:

Array inputs, with the frequencies set to low, medium and high respectively.

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

`pandas` dataframe, with the frequency set to "M" monthly.

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

## Finetuning

We have provided an example of finetuning the model on a new dataset in `notebooks/finetuning.ipynb`.

## Contribution Style guide

If you would like to submit a PR please make sure that you use our formatting style. We use [yapf](https://github.com/google/yapf) for formatting with the following options,

```
[style]
based_on_style = google
# Add your custom style rules here
indent_width = 2
spaces_before_comment = 2

```

Please run `yapf --in-place --recursive <filename>` on all affected files.
