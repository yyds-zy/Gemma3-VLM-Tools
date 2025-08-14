# 💬 Gemma-3-VLM Chat Apo

> 一个基于 Google Gemma-3-4B-IT 的中文对话示例，支持「文本 + 可选图片」输入，界面友好、开箱即用。

---

## ✨ 特性

- **一次加载，多次对话**：模型权重只在启动时加载，后续对话秒级响应  
- **支持图文混合**：可上传图片并配合文字提问  
- **中文优化**：系统提示默认中文，回答更贴近中文语境  
- **聊天记录管理**：一键清空 / 导出 JSON  
- **轻量界面**：基于 Gradio，浏览器即可访问，无需额外前端代码  

---

## 🚀 快速开始

### 1. 环境准备

```
pip install -U gradio transformers accelerate torch pillow
```

### 2. 下载模型权重
从官方或镜像站下载 gemma-3-4b-it
解压到本地任意目录，例如
D:\workspace\gemma-3-4b-it


### 3. 修改模型路径
打开 app.py，把 MODEL_ID 改成你的实际路径：
```
MODEL_ID = r"D:\workspace\gemma-3-4b-it"
```

### 4. 运行
```
python app.py
```

终端出现
Running on local URL:  http://127.0.0.1:7860
浏览器自动打开，即可开始对话。

---
## 🖥️ 界面说明
| 区域         | 功能                      |
| ---------- | ----------------------- |
| 左侧 Chatbot | 显示完整对话记录                |
| 右侧 Image   | 上传图片（可选）                |
| 右侧 Textbox | 输入文字问题                  |
| 发送 / 回车    | 提交问题                    |
| 清空历史       | 一键清空聊天记录                |
| 导出聊天记录     | 保存为 `chat_history.json` |

## 🛠️ 常见问题
| 问题       | 解决                                                                           |
| -------- | ---------------------------------------------------------------------------- |
| GPU 显存不足 | 将 `torch_dtype=torch.bfloat16` 改为 `torch.float16` 或 `torch.float32`；或仅使用 CPU |
| 中文乱码     | 确保终端 + 浏览器编码均为 UTF-8                                                         |
| 加载慢      | 首次需下载权重到缓存，耐心等待；后续秒开                                                         |
| 端口冲突     | 启动时加参数 `demo.launch(share=False, server_port=7861)`                          |

## 📜 License
代码遵循 MIT License，模型权重遵循 Google Gemma 官方协议。
## 🤝 贡献 & 反馈
欢迎提 Issue / PR，一起让对话体验更好！