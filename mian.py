# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple web interactive chat demo based on gradio, support Qwen1.5/2.5/3.0."""

from argparse import ArgumentParser
from threading import Thread
import gc
import os

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# 定义不同版本Qwen的默认路径（你可以根据实际路径修改）
DEFAULT_MODEL_PATHS = {
    "Qwen1.5": r"D:\\work\\py\\Qwen3\\Qwen1.5",
    "Qwen2.5": r"D:\\work\\py\\Qwen3\\Qwen2.5",
    "Qwen3.0": r"D:\\work\\py\\Qwen3\\Qwen3.0"
}

# 全局变量存储当前加载的模型/Tokenizer和版本
current_model = None
current_tokenizer = None
current_model_version = None


def _get_args():
    parser = ArgumentParser(description="Qwen series (1.5/2.5/3.0) Instruct web chat demo.")
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run demo with CPU only"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a publicly shareable link for the interface.",
    )
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        default=False,
        help="Automatically launch the interface in a new tab on the default browser.",
    )
    parser.add_argument(
        "--server-port", type=int, default=8000, help="Demo server port."
    )
    parser.add_argument(
        "--server-name", type=str, default="127.0.0.1", help="Demo server name."
    )

    args = parser.parse_args()
    return args


def _load_model_tokenizer(model_version, args):
    """根据选择的模型版本加载对应的模型和Tokenizer"""
    global current_model, current_tokenizer, current_model_version

    # 如果当前已加载相同版本，直接返回
    if current_model_version == model_version and current_model is not None:
        return current_model, current_tokenizer

    # 清理之前的模型
    _gc()

    # 获取模型路径
    checkpoint_path = DEFAULT_MODEL_PATHS[model_version]
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"模型路径不存在: {checkpoint_path}，请检查DEFAULT_MODEL_PATHS配置")

    print(f"正在加载 {model_version} 模型，路径: {checkpoint_path}")

    # 加载Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        resume_download=True,
        trust_remote_code=True  # Qwen系列需要开启这个参数
    )

    # 设置设备
    if args.cpu_only:
        device_map = "cpu"
        torch_dtype = torch.float32
    else:
        device_map = "auto"
        torch_dtype = "auto"

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        resume_download=True,
        trust_remote_code=True  # Qwen系列需要开启这个参数
    ).eval()

    # 设置生成参数
    model.generation_config.max_new_tokens = 2048
    model.generation_config.temperature = 0.7
    model.generation_config.top_p = 0.8

    # 更新全局变量
    current_model = model
    current_tokenizer = tokenizer
    current_model_version = model_version

    print(f"{model_version} 模型加载完成！")
    return model, tokenizer


def _chat_stream(model, tokenizer, query, history):
    """流式生成回复"""
    try:
        # 构建对话历史
        conversation = []
        for query_h, response_h in history:
            conversation.append({"role": "user", "content": query_h})
            conversation.append({"role": "assistant", "content": response_h})
        conversation.append({"role": "user", "content": query})

        # 应用聊天模板
        input_text = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        # 编码输入
        inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

        # 初始化流式生成器
        streamer = TextIteratorStreamer(
            tokenizer=tokenizer,
            skip_prompt=True,
            timeout=120.0,
            skip_special_tokens=True
        )

        # 启动生成线程
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "pad_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # 流式返回结果
        for new_text in streamer:
            yield new_text

    except Exception as e:
        yield f"生成回复时出错: {str(e)}"


def _gc():
    """清理内存/GPU缓存"""
    global current_model, current_tokenizer

    # 清理模型
    if current_model is not None:
        del current_model
        current_model = None
    if current_tokenizer is not None:
        del current_tokenizer
        current_tokenizer = None

    # 垃圾回收
    gc.collect()

    # 清理CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _launch_demo(args):
    """启动Gradio演示界面"""

    def load_selected_model(model_version):
        """加载选中的模型"""
        try:
            model, tokenizer = _load_model_tokenizer(model_version, args)
            return gr.update(value=f"✅ {model_version} 模型已加载完成！")
        except Exception as e:
            return gr.update(value=f"❌ 加载失败: {str(e)}")

    def predict(_query, _model_version, _chatbot, _task_history):
        """生成回复"""
        if not _query.strip():
            yield _chatbot
            return

        # 确保模型已加载
        if current_model_version != _model_version or current_model is None:
            try:
                model, tokenizer = _load_model_tokenizer(_model_version, args)
            except Exception as e:
                _chatbot.append((_query, f"加载模型失败: {str(e)}"))
                yield _chatbot
                return
        else:
            model, tokenizer = current_model, current_tokenizer

        print(f"[{_model_version}] User: {_query}")
        _chatbot.append((_query, ""))
        full_response = ""
        response = ""

        # 流式生成回复
        for new_text in _chat_stream(model, tokenizer, _query, history=_task_history):
            response += new_text
            _chatbot[-1] = (_query, response)
            yield _chatbot
            full_response = response

        # 更新历史记录
        _task_history.append((_query, full_response))
        print(f"[{_model_version}] Qwen: {full_response}")

    def regenerate(_model_version, _chatbot, _task_history):
        """重新生成回复"""
        if not _task_history:
            yield _chatbot
            return

        item = _task_history.pop(-1)
        _chatbot.pop(-1)
        yield from predict(item[0], _model_version, _chatbot, _task_history)

    def reset_user_input():
        """重置输入框"""
        return gr.update(value="")

    def reset_state(_chatbot, _task_history):
        """重置对话状态"""
        _task_history.clear()
        _chatbot.clear()
        _gc()
        return _chatbot

    with gr.Blocks(title="Qwen系列模型聊天演示") as demo:
        # 页面标题和说明
        gr.Markdown("""\
<p align="center"><img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/assets/logo/qwen2.5_logo.png" style="height: 120px"/><p>""")
        gr.Markdown(
            """\
<center><font size=4>Qwen系列模型 (1.5/2.5/3.0) 交互式聊天演示</center>"""
        )

        # 模型选择区域
        with gr.Row():
            with gr.Column(scale=3):
                model_version = gr.Dropdown(
                    choices=["Qwen1.5", "Qwen2.5", "Qwen3.0"],
                    value="Qwen2.5",
                    label="选择模型版本",
                    info="请确保对应版本的模型文件已下载到指定路径"
                )
            with gr.Column(scale=1):
                load_model_btn = gr.Button("📥 加载选中的模型", variant="primary")
                load_status = gr.Textbox(label="加载状态", value="未加载任何模型", interactive=False)

        # 聊天区域
        chatbot = gr.Chatbot(label="对话窗口", elem_classes="control-height", height=500)

        # 输入区域
        query = gr.Textbox(lines=3, label="输入你的问题", placeholder="请输入想要提问的内容...")

        # 状态存储
        task_history = gr.State([])

        # 功能按钮区域
        with gr.Row():
            empty_btn = gr.Button("🧹 清除历史", variant="secondary")
            submit_btn = gr.Button("🚀 发送", variant="primary")
            regen_btn = gr.Button("🔄 重新生成", variant="secondary")

        # 状态提示区域
        gr.Markdown("""\
<font size=2 color="gray">
注意事项：<br>
1. 首次加载模型可能需要较长时间，请耐心等待<br>
2. 切换模型版本后需要点击"加载选中的模型"按钮<br>
3. 建议使用GPU运行，CPU模式下速度较慢<br>
4. 本演示受Qwen系列模型的许可协议限制
</font>""")

        # 绑定事件
        load_model_btn.click(
            load_selected_model,
            inputs=[model_version],
            outputs=[load_status]
        )

        submit_btn.click(
            predict,
            inputs=[query, model_version, chatbot, task_history],
            outputs=[chatbot],
            show_progress=True
        )
        submit_btn.click(reset_user_input, [], [query])

        empty_btn.click(
            reset_state,
            inputs=[chatbot, task_history],
            outputs=[chatbot],
            show_progress=True
        )

        regen_btn.click(
            regenerate,
            inputs=[model_version, chatbot, task_history],
            outputs=[chatbot],
            show_progress=True
        )

    # 启动demo
    demo.queue(max_size=10).launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
        show_error=True
    )


def main():
    args = _get_args()
    _launch_demo(args)


if __name__ == "__main__":
    main()