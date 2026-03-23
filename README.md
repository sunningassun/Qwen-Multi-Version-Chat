# Qwen-Multi-Version-Chat

一个支持 Qwen1.5/2.5/3.0 多版本切换的交互式 Web 聊天演示工具，基于 Gradio 构建，简单易用，开箱即用。

## 📋 功能特点

- ✅ 支持 Qwen1.5/2.5/3.0 多版本模型切换
- ✅ 流式回复，实时显示生成内容
- ✅ 支持 CPU/GPU 两种运行模式
- ✅ 保留对话历史，支持重新生成 / 清空历史
- ✅ 简洁美观的 Web 界面，操作简单
- ✅ 自动清理内存 / GPU 缓存，避免资源泄露

## 🛠️ 环境搭建

### 1. 基础环境要求

- Python 3.8 及以上版本
- 足够的磁盘空间（模型文件大小参考：7B 约 14GB，14B 约 28GB）
- GPU 推荐（可选）：NVIDIA GPU + CUDA 11.7+（CPU 模式也可运行，但速度较慢）

### 2. 克隆代码仓库

```
git clone https://github.com/sunningassun/Qwen-Multi-Version-Chat.git
cd Qwen-Multi-Version-Chat
```

### 3. 安装依赖包

#### 方式 1：使用 pip 直接安装

```
# 基础依赖（必装）
pip install gradio transformers torch accelerate sentencepiece

# 如果使用GPU，建议安装对应版本的PyTorch（以CUDA 12.1为例）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 方式 2：使用 requirements.txt（推荐）

先创建 `requirements.txt` 文件：

```
gradio>=4.0.0
transformers>=4.37.0
torch>=2.0.0
accelerate>=0.20.0
sentencepiece>=0.1.99
protobuf>=4.25.0
```

然后执行安装：

```
pip install -r requirements.txt
```

### 4. 下载 Qwen 模型文件

你需要先下载对应版本的 Qwen 模型文件（Instruct 版本），支持以下来源：

#### 方式 1：从 ModelScope 下载

```
# 安装 modelscope
pip install modelscope

# 下载 Qwen2.5-7B-Instruct 示例（其他版本同理）
from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('qwen/Qwen2.5-7B-Instruct', cache_dir='./Qwen2.5')
```

#### 方式 2：从 Hugging Face 下载

直接从 [Qwen 官方 Hugging Face 仓库](https://huggingface.co/Qwen) 下载对应版本的模型文件，解压到本地目录。

#### 模型目录结构示例

```
Qwen-Multi-Version-Chat/
├── Qwen1.5/          # Qwen1.5 模型文件目录
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer.json
├── Qwen2.5/          # Qwen2.5 模型文件目录
└── Qwen3.0/          # Qwen3.0 模型文件目录
```

## 🚀 使用指南

### 1. 配置模型路径

打开 `chat_demo.py` 文件，修改 `DEFAULT_MODEL_PATHS` 字典，指向你本地的模型文件路径：

```
DEFAULT_MODEL_PATHS = {
    "Qwen1.5": r"D:\\work\\py\\Qwen3\\Qwen1.5",  # 替换为你的Qwen1.5路径
    "Qwen2.5": r"D:\\work\\py\\Qwen3\\Qwen2.5",  # 替换为你的Qwen2.5路径
    "Qwen3.0": r"D:\\work\\py\\Qwen3\\Qwen3.0"   # 替换为你的Qwen3.0路径
}
```

### 2. 启动应用

#### 基本启动（GPU 模式）

```
python chat_demo.py
```

#### CPU 模式启动（无 GPU 时使用）

```
python chat_demo.py --cpu-only
```

#### 高级启动选项

```
# 公开访问（生成公网链接）+ 自动打开浏览器
python chat_demo.py --share --inbrowser

# 指定端口和IP
python chat_demo.py --server-port 8080 --server-name 0.0.0.0
```

### 3. 界面操作

1. 启动后，浏览器会自动打开 `http://127.0.0.1:8000`（默认端口）

2. 在 "选择模型版本" 下拉框中选择要使用的 Qwen 版本

3. 点击 "📥 加载选中的模型" 按钮，等待模型加载完成（首次加载可能需要 1-2 分钟）

4. 在输入框中输入问题，点击 "🚀 发送" 按钮开始对话

5. 可选操作：

   - 🧹 清除历史：清空所有对话记录
   - 🔄 重新生成：重新生成上一轮的回复

   

## ⚙️ 命令行参数说明

|      参数       |           说明            |  默认值   |
| :-------------: | :-----------------------: | :-------: |
|  `--cpu-only`   | 使用 CPU 运行（禁用 GPU） |   False   |
|    `--share`    |   生成公网可访问的链接    |   False   |
|  `--inbrowser`  |    自动打开默认浏览器     |   False   |
| `--server-port` |        服务器端口         |   8000    |
| `--server-name` |        服务器地址         | 127.0.0.1 |

## ❓ 常见问题

### Q1: 模型加载失败怎么办？

- 检查模型路径是否正确，确保路径指向包含 `config.json`、`tokenizer.json` 的目录
- 确认模型文件完整，没有缺失或损坏
- 增加日志输出：在启动命令前添加 `PYTHONDEBUG=1`

### Q2: GPU 内存不足？

- 使用更小的模型版本（如 7B 而非 14B/32B）
- 设置 `device_map="auto"` 自动分配模型层
- 启用 CPU 模式：`--cpu-only`

### Q3: 回复生成速度慢？

- 确保使用 GPU 运行（CPU 模式速度会慢很多）
- 降低模型生成参数（如 `max_new_tokens` 调小）
- 关闭其他占用 GPU 的程序

### Q4: 中文乱码问题？

- 确保终端 / 浏览器编码为 UTF-8
- 升级 `transformers` 到最新版本：`pip install --upgrade transformers`

## 📝 注意事项

1. 模型加载需要一定时间，尤其是首次加载，请耐心等待
2. 切换模型版本后需要重新点击 "加载选中的模型" 按钮
3. 本工具仅用于学习和测试，请勿用于商业用途
4. 使用时请遵守 Qwen 模型的开源许可协议
5. 建议使用 16GB 以上内存，GPU 建议 10GB 以上显存（7B 模型）
