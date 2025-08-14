"""
Gemma-3-VLM Chat Demo
运行方式：
    pip install gradio transformers accelerate torch pillow
    python app.py
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

import gradio as gr
import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

# ==========================
# 1. 全局模型加载（只跑一次）
# ==========================
MODEL_ID = r"D:\workspace\gemma-3-4b-it"   # <- 改成你本地的路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("[INFO] 正在加载模型，请稍候……")
t0 = time.perf_counter()
model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map=DEVICE,
    torch_dtype=torch.bfloat16,
).eval()
model.forward = torch._dynamo.disable(model.forward)

processor = AutoProcessor.from_pretrained(MODEL_ID)
t1 = time.perf_counter()
print(f"[INFO] 模型加载完毕，耗时 {t1 - t0:.2f} s")
# ==========================
# 2. 对话函数
# ==========================
def build_messages(history: List[List[str]],
                   user_text: str,
                   pil_img: Image.Image | None) -> List[Dict]:
    """
    把 gradio history 转成 transformers 所需的 messages 格式
    history 中每一项现在是 [user_text, bot_text, pil_img | None]
    """
    messages = [{"role": "system",
                 "content": [{"type": "text", "text": "你是一个中文问答小助手。"}]}]

    # 逐条追加历史记录
    for human, assistant, _ in history:               # ### 改动：解包时忽略图片
        content = [{"type": "text", "text": human}]
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant",
                         "content": [{"type": "text", "text": assistant}]})

    # 最新用户输入
    final_content = []
    if pil_img is not None:
        final_content.append({"type": "image", "image": pil_img})
    final_content.append({"type": "text", "text": user_text})
    messages.append({"role": "user", "content": final_content})

    return messages


def chat_fn(history: List[List[str]],
            user_text: str,
            pil_img: Image.Image | None) -> Tuple[List[List[str]], str]:
    """
    history  -> [[user1, bot1, img1], [user2, bot2, img2], ...]
    user_text -> 本次输入文本
    pil_img   -> 本次上传图片（可选）

    返回 (新的 history, 状态信息)
    """
    if not user_text.strip():
        return history, "请输入文字内容！"

    messages = build_messages(history, user_text, pil_img)

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    t2 = time.perf_counter()
    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id
        )
        generation = generation[0][input_len:]
    t3 = time.perf_counter()

    decoded = processor.decode(generation, skip_special_tokens=True)

    # ### 改动：把本次图片一起存进历史
    history.append([user_text, decoded, pil_img])
    return history, f"推理耗时 {t3 - t2:.2f} s"

# ==========================
# 3. Gradio 界面
# ==========================
css = """
.gradio-container {max-width: 900px !important}
footer {visibility: hidden}
"""

with gr.Blocks(title="Gemma-3-VLM 对话助手", css=css, theme="soft") as demo:
    gr.Markdown("# 💬 Gemma-3-VLM 对话助手")
    gr.Markdown("上传图片（可选）并输入文字，即可与模型对话。模型只加载一次，后续对话秒级响应。")

    with gr.Row():
        with gr.Column(scale=3):
            # ### 改动：把 Chatbot 的 elem 属性打开，让它支持图片
            chatbot = gr.Chatbot(label="聊天记录", height=500, type="messages")
        with gr.Column(scale=1):
            image_box = gr.Image(type="pil", label="上传图片（可选）")
            text_box = gr.Textbox(
                placeholder="在此输入问题……",
                lines=2,
                label="文字输入"
            )
            btn_send = gr.Button("发送", variant="primary")
            status = gr.Textbox(label="状态", interactive=False)

    # 快捷按钮
    with gr.Row():
        btn_clear = gr.Button("清空历史")
        btn_export = gr.Button("导出聊天记录")

    # 事件绑定
    def clear_fn():
        return [], ""  # ### 改动：返回空列表，格式与 history 一致

    def export_fn(history):
        # 导出时去掉图片，避免 JSON 无法序列化 PIL 对象
        export_data = [[h[0], h[1]] for h in history]
        path = Path("chat_history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        return str(path.absolute())

    btn_send.click(chat_fn,
                   inputs=[chatbot, text_box, image_box],
                   outputs=[chatbot, status])
    btn_clear.click(clear_fn, outputs=[chatbot, status])
    btn_export.click(export_fn, inputs=chatbot, outputs=status)

    # 回车发送
    text_box.submit(chat_fn,
                    inputs=[chatbot, text_box, image_box],
                    outputs=[chatbot, status])

if __name__ == "__main__":
    demo.launch(share=False)