# pip install accelerate
import time
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

model_id = r"D:\workspace\gemma-3-4b-it"
# model_id = "google/gemma-3-27b-it"

# -------------------------------------------------
# 1. 模型加载耗时
# -------------------------------------------------
t0 = time.perf_counter()
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="cuda"
).eval()
model = model.eval()
model.forward = torch._dynamo.disable(model.forward)
t1 = time.perf_counter()
print(f"【模型加载耗时】 {(t1 - t0):.2f} s")

processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                # 注意：下面这行演示了原链接，但网络原因导致解析失败
                "image": "https://p1.ssl.qhmsg.com/t01de9944986a5b441e.jpg"
            },
            {"type": "text", "text": "请用中文描述图片里面的内容."}
        ]
    }
]

# -------------------------------------------------
# 可选：提示用户网页解析失败
# -------------------------------------------------
# 方式 A：直接 print
print("\n[提示] 由于网络或链接原因，图片 URL 解析失败，请检查链接合法性并重试。\n")

# 方式 B：raise 一个异常，强制中断（若你希望在未成功下载图片时停止）
# raise RuntimeError("图片下载失败，请确认网络及链接有效性后重试！")

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

# -------------------------------------------------
# 2. 推理耗时
# -------------------------------------------------
t2 = time.perf_counter()
with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
t3 = time.perf_counter()
print(f"【推理耗时】 {(t3 - t2):.2f} s")

decoded = processor.decode(generation, skip_special_tokens=True)
print("\n【生成结果】\n", decoded)