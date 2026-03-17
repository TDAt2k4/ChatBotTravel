## 🚀 Live Demo

[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-Demo-yellow)](https://huggingface.co/spaces/kerodat2004/Chat_Bot_Travel)

👉 Click vào badge để mở web demo
# Travel Chatbot RAG (Vietnam Tourism Assistant)

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
=>
Query Processing (normalize + intent + city)
=>
FAISS Vector Search
=>
Filter (city + intent + keyword)
=>
Score Threshold + Guardrails
=>
Response Generator (RAG)
