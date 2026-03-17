import gradio as gr
from src.chatbot_rag import rag_answer

def chat_process(message, history):
    msg = message.lower().strip()

    # ========================
    # GREETING / GOODBYE
    # ========================
    greetings = ["hi", "hello", "xin chào", "chào", "hey"]
    if any(g in msg for g in greetings):
        return "Xin chào! Mình là trợ lý du lịch 🇻🇳. Bạn muốn đi đâu chơi?"

    goodbyes = ["bye", "tạm biệt", "chào nhé", "goodbye"]
    if any(g in msg for g in goodbyes):
        return "Tạm biệt! Chúc bạn có chuyến đi thật vui vẻ "
    
    thanks = ["cảm ơn", "thanks", "thank you"]
    if any(w in msg for w in thanks):
        return "Không có gì. Bạn cần thêm địa điểm nào cứ hỏi mình nhé!"

    # ========================
    # FOLLOW-UP LOGIC
    # ========================
    full_query = message

    follow_up_words = ["cái", "nữa", "thứ", "đó", "kia", "link"]
    is_follow_up = (
        len(message.split()) < 5 and
        any(w in msg for w in follow_up_words)
    )

    if len(history) > 0 and is_follow_up:
        last_user_msg = ""
        for turn in reversed(history):
            if isinstance(turn, dict) and turn.get("role") == "user":
                last_user_msg = turn.get("content", "")
                break

        full_query = f"{last_user_msg} {message}"

    try:
        return rag_answer(full_query)
    except Exception as e:
        return f"Hệ thống đang bận: {str(e)}"


demo = gr.ChatInterface(
    fn=chat_process,
    title="TRỢ LÝ DU LỊCH VIỆT NAM",
    description="Tìm kiếm địa điểm du lịch nổi bật tại Việt Nam",
    examples=[
        "Bãi biển đẹp ở Khánh Hòa",
        "Hưng Yên có gì chơi?",
        "Gợi ý khu vui chơi"
    ]
)

if __name__ == "__main__":
    demo.launch()