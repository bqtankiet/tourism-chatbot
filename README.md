# Chatbot Du Lịch Việt Nam (RAG)

<p align="center">
  <a href="https://bqtankiet.github.io/tourism-chatbot/" target="_blank">
    <img src="https://img.shields.io/badge/🚀_LIVE_DEMO-WEB-orange?style=for-the-badge" />
  </a>
  &nbsp;&nbsp;
  <a href="https://huggingface.co/spaces/bqtankiet/travel-chatbot" target="_blank">
    <img src="https://img.shields.io/badge/🤗_HUGGING_FACE-INFERENCE-yellow?style=for-the-badge" />
  </a>
</p>

<p align="center">
  <i>Web UI demo • AI inference deployed on Hugging Face Spaces</i>
</p>

![image](/assets/image.png)
![image2](/assets/image2.png)

---

## 🚀 Giới thiệu

Đây là chatbot du lịch Việt Nam sử dụng **Retrieval-Augmented Generation (RAG)**, giúp trả lời chính xác các câu hỏi về địa điểm du lịch thông qua:

* Kết hợp **semantic search** và **keyword search**
* Truy xuất thông tin từ **knowledge base**
* Sinh câu trả lời bằng LLM

---

## 🧠 Triển khai Model (Hugging Face Space)

Model được deploy trên Hugging Face Space đóng gói trong Docker Container

### ⚙️ Cách hoạt động

```
User Request
   ↓
Hugging Face Space (Docker Container)
   ↓
Server (FastAPI)
   ↓
Retrieval (FAISS)
   ↓
LLM (Ollama - Qwen2)
   ↓
Response
```
---

## 🛠️ Công nghệ

* FastAPI
* Ollama (Qwen3:0.6B)
* FAISS
* SentenceTransformers (BAAI/bge-m3)
* underthesea
* Docker

---

## ⚡ Quick Start

```bash
git clone https://github.com/bqtankiet/tourism-chatbot.git
cd tourism-chatbot

pip install -r requirements.txt

ollama pull qwen3:0.6b
ollama serve

python serve.py
```




