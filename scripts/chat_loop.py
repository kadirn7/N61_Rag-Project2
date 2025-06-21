from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient

# Ayarlar
QDRANT_URL = "http://localhost:6333"
INSTR_COLLECTION = "n61_instructions"
PRODUCT_COLLECTION = "product_info"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GROQ_API_KEY = "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" # Gerçek API anahtarınızı buraya koyun
LLM_MODEL = "llama3-70b-8192"

# Türkçe cevap vermesi ve asistan rolünü üstlenmesi için özel prompt şablonu
TURKISH_PROMPT_TEMPLATE = """
Sen N61 Alışveriş Asistanısın. Görevin, sana verilen bağlamdaki ürün ve bilgileri kullanarak kullanıcının sorularını yanıtlamak ve ona yardımcı olmaktır.

Kurallar:
- Kesinlikle stok adedi, ürün sayısı veya "şu anda X ürün var" gibi ifadeler kullanma.
- Bir kategori sorulduğunda önce yalnızca bir ürünü kısaca tanıt, ardından "Bu tarzdan hoşlanır mısınız?" diye sor.
- Kullanıcı "beğenmedim / hoşlanmadım" derse, aynı kategoriden farklı bir ürünü öner.
- Ürün açıklamalarında fiyat, kumaş ve kullanım mevsimi gibi en fazla 2-3 özellik ver; liste yapma, kısa paragraf yeterli.
- Kampanya bilgisi bağlamda yoksa "Şu anda aktif kampanya bilgim yok" de.
- Beden sorularında ölçü iste, iade sorularında prosedürü açıkla.
- Soruları her zaman Türkçe yanıtla.
- Cevabın sadece verilen bağlama dayanmalı. Eğer bağlamda cevap yoksa, "Bu konuda bilgi bulamadım, üzgünüm." de.

Bağlam:
{context}

Soru: {question}

Cevap (Türkçe):
"""

TURKISH_PROMPT = PromptTemplate(
    template=TURKISH_PROMPT_TEMPLATE, input_variables=["context", "question"]
)

# 1. Embedding Modeli
# HuggingFace hub'ından modelimizi yüklüyoruz.
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={'device': 'cpu'} # CPU üzerinde çalışması için
)

# 2. Qdrant Client ve Vector Store'lar
# Langchain'in Qdrant entegrasyonunu kullanarak iki farklı koleksiyon için bağlantı kuruyoruz.
client = QdrantClient(url=QDRANT_URL)

instr_store = Qdrant(
    client=client,
    collection_name=INSTR_COLLECTION,
    embeddings=embeddings,
    content_payload_key="content",
)

product_store = Qdrant(
    client=client,
    collection_name=PRODUCT_COLLECTION,
    embeddings=embeddings,
    content_payload_key="content",
)

# 3. Retriever'lar
# Artık hibrit arama yapmadığımız için search_type'ı belirtmiyoruz (varsayılan dense).
instr_retriever = instr_store.as_retriever(search_kwargs={"k": 3})
product_retriever = product_store.as_retriever(search_kwargs={"k": 1})

# 4. Ensemble Retriever
# İki retriever'ı birleştirerek tek bir sorguyla iki koleksiyonda da arama yapmayı sağlıyoruz.
# Ağırlıklar, hangi retriever'ın sonuçlarının daha öncelikli olacağını belirler.
ensemble_retriever = EnsembleRetriever(
    retrievers=[instr_retriever, product_retriever],
    weights=[0.4, 0.6] # Ürün aramasına biraz daha fazla öncelik veriyoruz
)

# 5. Sohbet Geçmişi (Memory)
# Modelin önceki konuşmaları hatırlaması için hafıza mekanizması.
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key='result'
)

# 6. Groq LLM
llm = ChatGroq(
    temperature=0.2,
    model_name=LLM_MODEL,
    api_key=GROQ_API_KEY
)

# 7. RetrievalQA Zinciri
# Tüm parçaları bir araya getiren ana RAG zinciri.
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=ensemble_retriever,
    memory=memory,
    chain_type_kwargs={"prompt": TURKISH_PROMPT} # Özel Türkçe prompt'u burada zincire dahil ediyoruz
)

# 8. Sohbet Döngüsü
def chat_loop():
    print("🤖 Merhaba! Ben N61 Alışveriş Asistanı. Nasıl yardımcı olabilirim?")
    print("✋ Çıkmak için boş bırak ve ENTER'a bas.")
    while True:
        question = input("\nSen: ")
        if not question.strip():
            break

        try:
            response = qa_chain.invoke({"query": question})
            print(f"🤖: {response['result']}")
        except Exception as e:
            print(f"Bir hata oluştu: {e}")

if __name__ == "__main__":
    chat_loop() 