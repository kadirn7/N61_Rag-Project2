import os
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

# 1) Ayarlar
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
INSTR_COLLECTION = "n61_instructions"
PRODUCT_COLLECTION = "product_info"
# Daha stabil ve çok dilli desteği olan bir model kullanıyoruz.
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 2) Embedding modeli ve Qdrant Client
model = SentenceTransformer(EMBED_MODEL)
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# ————————————————————————————————
# A) Sorular & Cevaplar (Embedding + Qdrant)
# ————————————————————————————————
df_inst = pd.read_csv(os.path.join("..", "data", "instructions.csv"), quotechar='"')

# Kolon kontrolü
required_cols = {"question", "answer"}
assert required_cols.issubset(df_inst.columns), f"Eksik kolonlar: {required_cols - set(df_inst.columns)}"

# Boş veya eksik satırları atla
filtered_rows = df_inst.dropna(subset=["question", "answer"])

# Sadece `question` sütununu embed ediyoruz, çünkü arama bu sütuna göre yapılacak.
inst_vectors = model.encode(filtered_rows["question"].tolist(), show_progress_bar=True)

client.recreate_collection(
    collection_name=INSTR_COLLECTION,
    vectors_config=VectorParams(size=inst_vectors.shape[1], distance=Distance.COSINE),
)

points = [
    PointStruct(id=i, vector=inst_vectors[i].tolist(), payload={
        "content": row["question"] + " " + row["answer"],
        "question": row["question"],
        "answer": row["answer"],
        "question_type": row.get("question_type", "")
    })
    for i, row in filtered_rows.iterrows()
]
client.upsert(collection_name=INSTR_COLLECTION, points=points, wait=True)
print(f"[✓] {INSTR_COLLECTION} koleksiyonu yüklendi ({len(points)} kayıt).")


# ————————————————————————————————
# B) Ürün Bilgileri (Embedding + Qdrant)
# ————————————————————————————————
df_prod = pd.read_csv(os.path.join("..", "data", "sahte_urunler.csv")).dropna()

# Embed edilecek metne kategori bilgisini de ekliyoruz
prod_texts = (df_prod["kategori"] + " - " + 
              df_prod["ürün_adı"] + " - " +
              df_prod["marka"] + " - " +
              df_prod["fiyat"].astype(str) + " - " +
              df_prod["açıklama"] + " - Bedenler: " +
              df_prod["bedenler"]).tolist()
prod_vectors = model.encode(prod_texts, show_progress_bar=True)

client.recreate_collection(
    collection_name=PRODUCT_COLLECTION,
    vectors_config=VectorParams(size=prod_vectors.shape[1], distance=Distance.COSINE),
)

points = [
    PointStruct(id=i,
        vector=prod_vectors[i].tolist(),
        payload={
            "content": prod_texts[i],
            "ürün_adı": df_prod.iloc[i]["ürün_adı"],
            "kategori": df_prod.iloc[i]["kategori"],
            "marka": df_prod.iloc[i]["marka"],
            "fiyat": df_prod.iloc[i]["fiyat"],
            "açıklama": df_prod.iloc[i]["açıklama"],
            "bedenler": df_prod.iloc[i]["bedenler"],
            "link": df_prod.iloc[i]["link"],
        })
    for i in range(len(df_prod))
]
client.upsert(collection_name=PRODUCT_COLLECTION, points=points, wait=True)
print(f"[✓] {PRODUCT_COLLECTION} koleksiyonu yüklendi ({len(points)} kayıt).") 