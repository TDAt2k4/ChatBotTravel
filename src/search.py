import sys
import os
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import unicodedata
from data.city_alias import get_city_alias

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# =========================
# LOAD DATA
# =========================
data_path = os.path.join(BASE_DIR, "dataset", "vietnam_travel_final.csv")
index_path = os.path.join(BASE_DIR, "vector_db", "travel_index.faiss")

data = pd.read_csv(data_path)
index = faiss.read_index(index_path)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# =========================
# NORMALIZE
# =========================
def normalize_text(text):
    text = str(text).strip().lower()
    text = text.replace("đ", "d")
    text = unicodedata.normalize("NFKD", text)
    text = "".join([c for c in text if not unicodedata.combining(c)])
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_meaningless_query(query):
    query = query.strip().lower()

    # quá ngắn
    if len(query) < 3:
        return True

    # không có chữ cái
    if not re.search(r"[a-zA-Zà-ỹ]", query):
        return True

    words = query.split()

    #  nếu 1 từ dài bất thường → spam
    if len(words) == 1 and len(words[0]) > 12:
        return True

    #  không chứa từ khóa du lịch nào
    tourism_keywords = [
        "đi", "du lịch", "chơi", "ăn", "ở", "đâu",
        "biển", "núi", "checkin", "khách sạn", "quán"
    ]

    if not any(k in query for k in tourism_keywords):
        return True

    return False

def clean_city_name(text):
    text = text.replace("tp ", "")
    text = text.replace("thanh pho ", "")
    return text.strip()

# =========================
# MATCH KEYWORD TIẾNG VIỆT
# =========================
def match_keywords_vn(text, keyword_list):
    for k in keyword_list:
        pattern = r"\b" + re.escape(k) + r"\b"
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

# =========================
# GUARDRAIL (GIỮ NHƯNG KHÔNG TRẢ ERROR)
# =========================
def check_guardrails(query):
    q = query.lower()

    off_topic_keywords = [
        "code", "toán", "bitcoin", "chính trị",
        "bài tập", "game", "tiếng anh"
    ]

    if any(word in q for word in off_topic_keywords):
        return "[OFF_TOPIC]"
    elif any(word in q for word in ["chào", "hi", "hello"]):
        return "[GREETING]"

    return "[TOURISM]"

# =========================
# INTENT
# =========================
def detect_intent(query):
    q = query.lower()

    if any(k in q for k in ["biển","bien","đảo","dao"]): return "beach"
    if any(k in q for k in ["ăn","food","quán","nhà hàng"]): return "food"
    if any(k in q for k in ["checkin","sống ảo","đẹp"]): return "checkin"
    if any(k in q for k in ["chùa","di tích","lịch sử"]): return "culture"
    if any(k in q for k in ["núi","thác","rừng"]): return "nature"
    if any(k in q for k in ["chơi","giải trí","bar"]): return "entertainment"

    return "general"

# =========================
# CITY DETECTION
# =========================
raw_alias = get_city_alias()
city_alias = {
    normalize_text(k): clean_city_name(normalize_text(v))
    for k, v in raw_alias.items()
}

cities = [
    clean_city_name(normalize_text(c))
    for c in data["city"].dropna().unique()
]

def detect_city(query):
    for alias, city in city_alias.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", query):
            return city

    for city in cities:
        if re.search(r"\b" + re.escape(city) + r"\b", query):
            return city

    return None

# =========================
# MAIN SEARCH
# =========================
def search_places(query, top_k=3):
    print("\n==================================")
    print(f"USER QUERY: '{query}'")

    if is_meaningless_query(query):
        print("[!] Query vô nghĩa")
        return []

    routing = check_guardrails(query)

    #  KHÔNG return error nữa → để RAG xử lý
    if routing == "[OFF_TOPIC]":
        return []

    query_norm = normalize_text(query)
    detected_city = detect_city(query_norm)
    intent = detect_intent(query)

    print(f"[*] City: {detected_city}")
    print(f"[*] Intent: {intent}")

    # =========================
    # EMBEDDING SEARCH
    # =========================
    search_query = query_norm
    if detected_city:
        search_query = f"{search_query} {detected_city} {detected_city}"

    query_vector = model.encode([search_query])
    query_vector = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)

    D, I = index.search(query_vector.astype("float32"), 100)

    results = []
    used_indices = set()

    # =========================
    # FILTER + SCORE
    # =========================
    MIN_SCORE = 0.5

    for rank, i in enumerate(I[0]):
        place = data.iloc[i]
        score = float(D[0][rank])


        if score < MIN_SCORE:
            continue

        place_city = clean_city_name(normalize_text(place["city"]))

        name_raw = str(place.get("name_raw", place.get("name", "")))
        desc_raw = str(place.get("description_raw", place.get("description", "")))
        text_vn = f"{name_raw} {desc_raw}".lower()

        if detected_city and place_city != detected_city:
            continue

        # intent filter
        if intent == "beach" and not match_keywords_vn(text_vn, ["biển","đảo","bãi"]): continue
        if intent == "food" and not match_keywords_vn(text_vn, ["ăn","quán","nhà hàng"]): continue
        if intent == "checkin" and not match_keywords_vn(text_vn, ["đẹp","check"]): continue
        if intent == "culture" and not match_keywords_vn(text_vn, ["chùa","di tích"]): continue
        if intent == "nature" and not match_keywords_vn(text_vn, ["núi","rừng","thác"]): continue

        results.append((place, score))
        used_indices.add(i)

        if len(results) >= top_k:
            break

    # =========================
    # FALLBACK NHẸ (GIỮ)
    # =========================
    if len(results) == 0:
        return []

    # =========================
    # FORMAT OUTPUT + SCORE
    # =========================
    final = []
    for place, score in results:
        final.append({
            "name": str(place.get("name_raw", place.get("name", ""))),
            "city": str(place.get("city_raw", place.get("city", ""))),
            "description": str(place.get("description_raw", place.get("description", ""))),
            "maps_link": str(place.get("maps_link", "")),
            "score": score
        })

    return final