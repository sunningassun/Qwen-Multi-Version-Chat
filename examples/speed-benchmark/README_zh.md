# 效率评估

本文介绍Qwen2.5系列模型（原始模型和量化模型）的效率测试流程，详细报告可参考 [Qwen2.5模型效率评估报告](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html)。

## 1. 模型资源

对于托管在HuggingFace上的模型，可参考 [Qwen2.5模型-HuggingFace](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)。

对于托管在ModelScope上的模型，可参考 [Qwen2.5模型-ModelScope](https://modelscope.cn/collections/Qwen25-dbc4d30adb768)。


## 2. 环境安装

使用HuggingFace transformers推理，安装环境如下：

```shell
conda create -n qwen_perf_transformers python=3.10
conda activate qwen_perf_transformers

pip install torch==2.3.1
pip install git+https://github.com/AutoGPTQ/AutoGPTQ.git@v0.7.1
pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.5.8
pip install -r requirements-perf-transformers.txt
```

> [!Important]
> - 对于 `flash-attention`，您可以从 [GitHub 发布页面](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.5.8) 使用预编译的 wheel 包进行安装，或者从源代码安装，后者需要一个兼容的 CUDA 编译器。
>   - 实际上，您并不需要单独安装 `flash-attention`。它已经被集成到了 `torch` 中作为 `sdpa` 的后端实现。
> - 若要使 `auto_gptq` 使用高效的内核，您需要从源代码安装，因为预编译的 wheel 包依赖于与之不兼容的 `torch` 版本。从源代码安装同样需要一个兼容的 CUDA 编译器。
> - 若要使 `autoawq` 使用高效的内核，您需要安装 `autoawq-kernels`，该组件应当会自动安装。如果未自动安装，请运行 `pip install autoawq-kernels` 进行手动安装。


使用vLLM推理，安装环境如下：

```shell
conda create -n qwen_perf_vllm python=3.10
conda activate qwen_perf_vllm

pip install -r requirements-perf-vllm.txt
```


## 3. 执行测试

下面介绍两种执行测试的方法，分别是使用脚本测试和使用Speed Benchmark工具进行测试。

### 方法1：使用Speed Benchmark工具测试

使用[EvalScope](https://github.com/modelscope/evalscope)开发的Speed Benchmark工具进行测试，支持自动从modelscope下载模型并输出测试结果，也支持指定模型服务的url进行测试，具体请参考[📖使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/speed_benchmark.html)。

**安装依赖**
```shell
pip install 'evalscope[perf]' -U
```

#### HuggingFace transformers推理

执行命令如下：
```shell
CUDA_VISIBLE_DEVICES=0 evalscope perf \
 --parallel 1 \
 --Qwen3.0 Qwen/Qwen2.5-0.5B-Instruct \
 --attn-implementation flash_attention_2 \
 --log-every-n-query 5 \
 --connect-timeout 6000 \
 --read-timeout 6000 \
 --max-tokens 2048 \
 --min-tokens 2048 \
 --api local \
 --dataset speed_benchmark 
```

#### vLLM推理

```shell
CUDA_VISIBLE_DEVICES=0 evalscope perf \
 --parallel 1 \
 --Qwen3.0 Qwen/Qwen2.5-0.5B-Instruct \
 --log-every-n-query 1 \
 --connect-timeout 60000 \
 --read-timeout 60000\
 --max-tokens 2048 \
 --min-tokens 2048 \
 --api local_vllm \
 --dataset speed_benchmark
```

#### 参数说明
- `--parallel` 设置并发请求的worker数量，需固定为1。
- `--model` 测试模型文件路径，也可为模型ID，支持自动从modelscope下载模型，例如Qwen/Qwen2.5-0.5B-Instruct。
- `--attn-implementation` 设置attention实现方式，可选值为flash_attention_2|eager|sdpa。
- `--log-every-n-query`: 设置每n个请求打印一次日志。
- `--connect-timeout`: 设置连接超时时间，单位为秒。
- `--read-timeout`: 设置读取超时时间，单位为秒。
- `--max-tokens`: 设置最大输出长度，单位为token。
- `--min-tokens`: 设置最小输出长度，单位为token；两个参数同时设置为2048则模型固定输出长度为2048。
- `--api`: 设置推理接口，本地推理可选值为local|local_vllm。
- `--dataset`: 设置测试数据集，可选值为speed_benchmark|speed_benchmark_long。

#### 测试结果

测试结果详见`outputs/{model_name}/{timestamp}/speed_benchmark.json`文件，其中包含所有请求结果和测试参数。

### 方法2：使用脚本测试

#### HuggingFace transformers推理

- 使用HuggingFace hub

```shell
python speed_benchmark_transformers.py --model_id_or_path Qwen/Qwen2.5-0.5B-Instruct --context_length 1 --gpus 0 --outputs_dir outputs/transformers

# 指定HF_ENDPOINT
HF_ENDPOINT=https://hf-mirror.com python speed_benchmark_transformers.py --model_id_or_path Qwen/Qwen2.5-0.5B-Instruct --context_length 1 --gpus 0 --outputs_dir outputs/transformers
```

- 使用ModelScope hub

```shell
python speed_benchmark_transformers.py --model_id_or_path Qwen/Qwen2.5-0.5B-Instruct --context_length 1 --gpus 0 --use_modelscope --outputs_dir outputs/transformers
```

参数说明：

    `--model_id_or_path`: 模型ID或本地路径， 可选值参考`模型资源`章节  
    `--context_length`: 输入长度，单位为token数；可选值为1, 6144, 14336, 30720, 63488, 129024；具体可参考`Qwen2.5模型效率评估报告`  
    `--generate_length`: 生成token数量；默认为2048
    `--gpus`: 等价于环境变量CUDA_VISIBLE_DEVICES，例如`0,1,2,3`，`4,5`  
    `--use_modelscope`: 如果设置该值，则使用ModelScope加载模型，否则使用HuggingFace  
    `--outputs_dir`: 输出目录， 默认为`outputs/transformers`  


#### vLLM推理

- 使用HuggingFace hub

```shell
python speed_benchmark_vllm.py --model_id_or_path Qwen/Qwen2.5-0.5B-Instruct --context_length 1 --max_model_len 32768 --gpus 0 --gpu_memory_utilization 0.9 --outputs_dir outputs/vllm

# 指定HF_ENDPOINT
HF_ENDPOINT=https://hf-mirror.com python speed_benchmark_vllm.py --model_id_or_path Qwen/Qwen2.5-0.5B-Instruct --context_length 1 --max_model_len 32768 --gpus 0 --gpu_memory_utilization 0.9 --outputs_dir outputs/vllm
```

- 使用ModelScope hub

```shell
python speed_benchmark_vllm.py --model_id_or_path Qwen/Qwen2.5-0.5B-Instruct --context_length 1 --max_model_len 32768 --gpus 0 --use_modelscope --gpu_memory_utilization 0.9 --outputs_dir outputs/vllm
```

参数说明：

    `--model_id_or_path`: 模型ID或本地路径， 可选值参考`模型资源`章节  
    `--context_length`: 输入长度，单位为token数；可选值为1, 6144, 14336, 30720, 63488, 129024；具体可参考`Qwen2.5模型效率评估报告`  
    `--generate_length`: 生成token数量；默认为2048
    `--max_model_len`: 模型最大长度，单位为token数；默认为32768  
    `--gpus`: 等价于环境变量CUDA_VISIBLE_DEVICES，例如`0,1,2,3`，`4,5`   
    `--use_modelscope`: 如果设置该值，则使用ModelScope加载模型，否则使用HuggingFace  
    `--gpu_memory_utilization`: GPU内存利用率，取值范围为(0, 1]；默认为0.9  
    `--outputs_dir`: 输出目录， 默认为`outputs/vllm`  
    `--enforce_eager`: 是否强制使用eager模式；默认为False  

#### 测试结果

测试结果详见`outputs`目录下的文件，默认包括`transformers`和`vllm`两个目录，分别存放HuggingFace transformers和vLLM的测试结果。

## 注意事项

1. 多次测试，取平均值，典型值为3次
2. 测试前请确保GPU处于空闲状态，避免其他任务影响测试结果


