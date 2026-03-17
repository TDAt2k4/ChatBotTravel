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

from huggingface_hub import InferenceClient


def respond(
    message,
    history: list[dict[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
    hf_token: gr.OAuthToken,
):
    """
    For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
    """
    client = InferenceClient(token=hf_token.token, model="openai/gpt-oss-20b")

    messages = [{"role": "system", "content": system_message}]

    messages.extend(history)

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        choices = message.choices
        token = ""
        if len(choices) and choices[0].delta.content:
            token = choices[0].delta.content

        response += token
        yield response


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
chatbot = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)

with gr.Blocks() as demo:
    with gr.Sidebar():
        gr.LoginButton()
    chatbot.render()


if __name__ == "__main__":
    demo.launch()
