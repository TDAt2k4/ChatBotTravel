import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.search import search_places, is_meaningless_query

# ========================
# MODEL
# ========================
last_search_results = []
waiting_for_suggestion = False
last_query_no_result = ""

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# ========================
# MEMORY (tạm thời vẫn dùng global)
# ========================
last_search_results = []

# ========================
# PARAPHRASE 
# ========================
def paraphrase_description(text):
    messages = [
        {
            "role": "system",
            "content": (
                "Bạn chỉ có nhiệm vụ viết lại câu văn cho tự nhiên hơn.\n"
                "KHÔNG thêm thông tin mới.\n"
                "KHÔNG suy diễn.\n"
                "Giữ nguyên ý nghĩa 100%.\n"
                "Viết ngắn gọn, dễ hiểu."
            )
        },
        {
            "role": "user",
            "content": f"Viết lại câu sau:\n{text}"
        }
    ]

    text_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        top_p=0.8
    )

    return tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1]:],
        skip_special_tokens=True
    ).strip()


# ========================
# BUILD ANSWER
# ========================
def build_natural_answer(results):
    answer = ""
    for idx, p in enumerate(results, 1):
        desc = paraphrase_description(p["description"])
        answer += f"{idx}. {p['name']} ({p['city']}): {desc}\n"
    return answer

# ========================
# INTENT
# ========================
def detect_intent(query):
    q = query.lower()

    if any(k in q for k in ["biển","bien","beach","đảo","dao"]): return "beach"
    if any(k in q for k in ["ăn","food","quán","nhà hàng"]): return "food"
    if any(k in q for k in ["checkin","sống ảo","đẹp"]): return "checkin"
    if any(k in q for k in ["chùa","di tích","lịch sử"]): return "culture"
    if any(k in q for k in ["núi","thác","rừng"]): return "nature"
    if any(k in q for k in ["chơi","giải trí","bar"]): return "entertainment"

    return "general"

# ========================
# EMBEDDING CLASSIFIER
# ========================
def classify_query(results, threshold=1.2):
    if not results:
        return "no_data"

    if isinstance(results, list) and "error" in results[0]:
        return "no_data"

    top_score = results[0].get("score", 999)

    if top_score > threshold:
        return "out_domain"

    return "in_domain"

# ========================
# MAIN RAG
# ========================
def rag_answer(query):
    global last_search_results, waiting_for_suggestion, last_query_no_result

    query_lower = query.lower().strip()

    # ========================
    # 1. HANDLE GỢI Ý
    # ========================
    if waiting_for_suggestion:
        if any(x in query_lower for x in ["có","ok","yes","ừ"]):
            waiting_for_suggestion = False
            results = search_places("du lịch nổi bật việt nam")

            if results:
                last_search_results = results
                return "Gợi ý cho bạn:\n\n" + build_natural_answer(results)

            return "Chưa có gợi ý phù hợp."

        elif any(x in query_lower for x in ["không","no","ko"]):
            waiting_for_suggestion = False
            return "Ok, bạn cần gì cứ hỏi mình nhé!"

        else:
            return "Bạn có muốn mình gợi ý địa điểm khác không? (có / không)"

    # ========================
    # 2. HỎI LINK
    # ========================
    if "link" in query_lower and last_search_results:
        nums = re.findall(r'\d+', query)
        if nums:
            idx = int(nums[0]) - 1
            if 0 <= idx < len(last_search_results):
                p = last_search_results[idx]
                return f"Link: {p['maps_link']}"

    # ========================
    # 3. SEARCH
    # ========================
    intent = detect_intent(query)

    intent_map = {
        "beach": "du lịch biển",
        "food": "ăn uống",
        "checkin": "checkin đẹp",
        "culture": "văn hóa",
        "nature": "thiên nhiên",
        "entertainment": "giải trí",
        "general": ""
    }

    augmented_query = query + " " + intent_map.get(intent, "")

    results = search_places(augmented_query)

    # ========================
    # 4. CLASSIFY
    # ========================
    query_type = classify_query(results)

    if query_type == "out_domain":
        return "Mình chỉ hỗ trợ tư vấn du lịch (địa điểm, ăn uống, vui chơi)."

    if query_type == "no_data":
        #  nếu query vô nghĩa
        if is_meaningless_query(query):
            return "Mình chưa hiểu ý bạn. Mình chỉ hỗ trợ gợi ý địa điểm du lịch nhé!"

        #  nếu có nghĩa nhưng không có dữ liệu
        waiting_for_suggestion = True
        last_query_no_result = query
        return "Mình chưa có dữ liệu. Bạn có muốn mình gợi ý địa điểm khác không?"

    # ========================
    # 5. BUILD
    # ========================
    last_search_results = results
    return build_natural_answer(results)