# 🧳 Travel Chatbot RAG (Vietnam Tourism Assistant)

Chatbot AI hỗ trợ người dùng tìm kiếm và gợi ý địa điểm du lịch tại Việt Nam bằng kỹ thuật **RAG (Retrieval-Augmented Generation)**.

---

## Demo Features

-  Tìm kiếm địa điểm bằng **FAISS + Embedding**
-  Nhận diện **thành phố (city detection)**
- Phân loại **ý định người dùng (intent detection)**
  - Biển
  - Ăn uống
  - Check-in
  - Văn hóa
  - Thiên nhiên
-  Xử lý **query vô nghĩa (nonsense query)**
-  Gợi ý lại khi không có dữ liệu
- Trả link Google Maps cho từng địa điểm

---

##  System Architecture
User Query
↓
Query Processing (normalize + intent + city)
↓
FAISS Vector Search
↓
Filter (city + intent + keyword)
↓
Score Threshold + Guardrails
↓
Response Generator (RAG)
=======
---
title: Chat Bot Travel
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
- inference-api
---

An example chatbot using [Gradio](https://gradio.app), [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/v0.22.2/en/index), and the [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index).

