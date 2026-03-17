import pandas as pd
import urllib.parse
import unicodedata
import re


# =========================
# 1. NORMALIZE TEXT (CHO EMBEDDING)
# =========================
def normalize_text(text):
    text = str(text).strip().lower()

    # FIX quan trọng: đ → d
    text = text.replace("đ", "d")

    # bỏ dấu
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])

    # bỏ ký tự đặc biệt
    text = re.sub(r"[^\w\s]", " ", text)

    # bỏ khoảng trắng dư
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================
# 2. CLEAN KEYWORDS
# =========================
def clean_keywords(text):
    text = str(text).lower()
    text = text.replace('"', '')

    keywords = [k.strip() for k in text.split(",") if k.strip()]

    # FIX: normalize luôn keywords
    keywords = [normalize_text(k) for k in keywords]

    return " ".join(keywords)


# =========================
# 3. LOAD DATA
# =========================
df = pd.read_excel("dataset/DataSet.xlsx")

# rename cột
df = df.rename(columns={
    "Tên địa điểm": "name",
    "Vị trí": "city",
    "Mô tả": "description",
    "Đánh giá ": "rating",
    "Từ Khóa": "keywords"
})

# drop cột không cần
df = df.drop(columns=["STT", "Ảnh"], errors="ignore")

# drop null
df = df.dropna()

# =========================
# 4. GIỮ RAW TEXT (CHO HIỂN THỊ + MAPS)
# =========================
df["name_raw"] = df["name"].astype(str).str.strip()
df["city_raw"] = df["city"].astype(str).str.strip()
df["description_raw"] = df["description"].astype(str).str.strip()

# =========================
# 5. CLEAN DATA (CHO EMBEDDING)
# =========================
df["name"] = df["name"].apply(normalize_text)
df["city"] = df["city"].apply(normalize_text)
df["description"] = df["description"].apply(normalize_text)
df["keywords"] = df["keywords"].apply(clean_keywords)

# =========================
# 6. CONTENT CHO EMBEDDING
# =========================
df["content"] = (
    df["name"] + " " +
    df["city"] + " " +
    df["description"] + " " +
    df["keywords"] + " " +
    df["keywords"]   # boost keyword
)

# =========================
# 7. GOOGLE MAP LINK (DÙNG RAW + ENCODE)
# =========================
def create_maps_link(row):
    query = f"{row['name_raw']} {row['city_raw']}"
    encoded = urllib.parse.quote(query)
    return f"https://www.google.com/maps/search/?api=1&query={encoded}"

df["maps_link"] = df.apply(create_maps_link, axis=1)

# =========================
# 8. REMOVE DUPLICATE
# =========================
df = df.drop_duplicates(subset="name")

# =========================
# 9. SAVE FINAL
# =========================
df.to_csv("dataset/vietnam_travel_final.csv", index=False)

# =========================
# 10. DEBUG
# =========================
print(" Preprocessing done")
print("Total places:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nSample data:")
print(df.head(5))