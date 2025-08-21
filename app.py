import os
import json
import time
import base64
import io
from pathlib import Path
from typing import List, Dict, Tuple
from bs4 import BeautifulSoup

import gradio as gr
import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

# ==========================
# 1. 全局模型加载（只跑一次）
# ==========================
MODEL_ID = r"D:\workspace\gemma-3-4b-it"  # <- 改成你本地的路径
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
    """
    messages = [{"role": "system",
                 "content": [{"type": "text", "text": "你是一个中文问答小助手。"}]}]

    # 逐条追加历史记录
    for human, assistant in history:
        # 从可能包含HTML的历史消息中提取纯文本
        soup = BeautifulSoup(human, 'html.parser')
        user_content_text = soup.get_text(separator=' ', strip=True)
        content = [{"type": "text", "text": user_content_text}]
        messages.append({"role": "user", "content": content})

        # assistant 回合
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
    history -> [[user1, bot1], [user2, bot2], ...]
    user_text -> 本次输入文本
    pil_img  -> 本次上传图片（可选）

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

    # 将图片编码为 Base64 格式，并构建更清晰的 HTML
    user_msg_with_img = user_text
    if pil_img is not None:
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        img_html = f'<img src="data:image/png;base64,{img_str}" style="max-width: 100%; height: auto; border-radius: 8px;">'
        # 使用简单的 p 标签包装文本，增强可读性
        user_msg_with_img = f'{img_html}<p style="margin: 0; padding-top: 10px;">{user_text}</p>'


    # 追加到历史
    history.append([user_msg_with_img, decoded])
    return history, f"推理耗时 {t3 - t2:.2f} s"

# ==========================
# 3. Gradio 界面
# ==========================
css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto;
}
footer {
    visibility: hidden;
}
.gradio-row {
    flex-wrap: nowrap !important;
}
.gradio-column {
    min-width: 300px;
}
/* 优化 Chatbot 消息气泡的样式 */
.message-bubble {
    white-space: pre-wrap; /* 保持换行 */
    word-wrap: break-word; /* 强制单词换行 */
    overflow-x: auto; /* 防止水平溢出 */
}
.message-bubble img {
    max-width: 100%;
}
"""

with gr.Blocks(title="Gemma-3-VLM 对话助手", css=css, theme="soft") as demo:
    gr.Markdown("# 💬 Gemma-3-VLM 对话助手")
    gr.Markdown("上传图片（可选）并输入文字，即可与模型对话。模型只加载一次，后续对话秒级响应。")

    # 主体左右分栏
    with gr.Row():
        # 左侧：聊天记录
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="聊天记录", height="75vh", elem_classes="message-bubble")
        
        # 右侧：输入区域和图片
        with gr.Column(scale=1):
            image_box = gr.Image(type="pil", label="上传图片（可选）", interactive=True)
            text_box = gr.Textbox(
                placeholder="在此输入问题……",
                lines=2,
                label="文字输入"
            )
            btn_send = gr.Button("发送", variant="primary")
            status = gr.Textbox(label="状态", interactive=False, container=False)
    
    # 底部按钮
    with gr.Row():
        btn_clear = gr.Button("清空历史")
        btn_export = gr.Button("导出聊天记录")

    # 事件绑定
    def clear_fn():
        return [], None, ""

    def export_fn(history):
        path = Path("chat_history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        return str(path.absolute())

    btn_send.click(chat_fn,
                   inputs=[chatbot, text_box, image_box],
                   outputs=[chatbot, status])
    btn_clear.click(clear_fn, outputs=[chatbot, image_box, status])
    btn_export.click(export_fn, inputs=chatbot, outputs=status)

    # 回车发送
    text_box.submit(chat_fn,
                    inputs=[chatbot, text_box, image_box],
                    outputs=[chatbot, status])

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)