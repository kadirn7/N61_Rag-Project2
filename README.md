# N61-RAG-Project

Bu proje, bir e-ticaret platformu olan "N61" için oluşturulmuş, **Retrieval-Augmented Generation (RAG)** tabanlı bir sohbet robotudur. Kullanıcıların sıkça sorduğu sorulara yanıt vermek ve sipariş iade kodlarını sağlamak üzere tasarlanmıştır.

## ✨ Özellikler

- **Dinamik Cevaplar**: Kullanıcı sorularına, `data/instructions.csv` dosyasındaki bilgi tabanını kullanarak anında ve bağlama uygun cevaplar üretir.
- **Sipariş Yönetimi**: Kullanıcıların sipariş numaralarını kullanarak `data/order_returns.csv` dosyasından iade kodlarını sorgulamasına olanak tanır.
- **Genişletilebilir**: `data` klasöründeki CSV dosyalarına yeni veriler ekleyerek botun bilgi tabanını kolayca genişletebilirsiniz.
- **Modern Teknoloji Yığını**: Güçlü ve hızlı bir altyapı için Groq (LLM), Qdrant (vektör veritabanı) ve Sentence-Transformers kullanır.

## 📁 Proje Yapısı

```
your_project/
├── data/
│   ├── instructions.csv      # Genel sorular ve cevaplar
│   └── order_returns.csv     # Sipariş numaraları ve iade kodları
├── scripts/
│   ├── upload_to_qdrant.py   # Verileri işleyip Qdrant'a yükler
│   └── chat_loop.py          # Etkileşimli sohbeti başlatır
└── requirements.txt          # Gerekli Python kütüphaneleri
```

## 🚀 Kurulum ve Çalıştırma

### 1. Ön Gereksinimler

- Python 3.8+
- Docker (Qdrant'ı çalıştırmak için)

### 2. Qdrant'ı Başlatma

Terminalde aşağıdaki komut ile Qdrant'ı bir Docker container'ı olarak başlatın:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 3. Bağımlılıkları Yükleme

Proje için gerekli Python kütüphanelerini yükleyin:

```bash
pip install -r requirements.txt
```

### 4. Veri Tabanını Doldurma

`data/` klasöründeki CSV dosyalarının içeriğini işleyip vektör veritabanına yüklemek için aşağıdaki script'i çalıştırın:

```bash
python scripts/upload_to_qdrant.py
```
Bu komut, `n61_instructions` ve `order_returns` adında iki adet koleksiyonu Qdrant üzerinde oluşturacaktır.

### 5. Sohbeti Başlatma

Her şey hazır! Aşağıdaki komutla sohbet botunu başlatabilir ve sorularınızı sormaya başlayabilirsiniz:

```bash
python scripts/chat_loop.py
```

Sohbeti sonlandırmak için soru sormadan doğrudan `ENTER` tuşuna basmanız yeterlidir. 

## 🔄 Veri ve Akış Mantığı

- **instructions.csv**: Sıkça sorulan sorular, genel alışveriş süreçleri, iade, kampanya, teknik destek gibi konularda örnek soru-cevap çiftlerini içerir. Buradaki cevaplar, ürün önerisi gibi spesifik durumlarda asla belirli bir ürün adı içermez. Bunun yerine, kullanıcıdan kategori veya tercih soran, yönlendirici ve bağlamsal cevaplar bulunur (ör: "Bana ürün önerir misin?" sorusuna "Hangi kategoride ürün bakmıştınız?" gibi).
- **sahte_urunler.csv**: Gerçek ürün veritabanı gibi davranır. Ürün adı, kategori, açıklama, fiyat, marka, bedenler ve link gibi alanları içerir. Qdrant'a yüklenir ve ürün önerisi gerektiğinde asistan, buradaki ürünlerden uygun olanı (veya rastgele) önerir.
- **Ezberli Ürün Önerisi Yok**: instructions.csv'de doğrudan ürün adı geçen, "X ürünü öneririm" gibi cevaplar kaldırılmıştır. Asistan, ürün önerisi sorularında Qdrant'ın ürün koleksiyonunu kullanarak gerçek ürünlerden öneri yapar. Böylece sistem, yeni ürünler eklendiğinde veya ürünler değiştiğinde otomatik olarak güncel öneriler sunar.
- **Kategorik ve Yönlendirici Akış**: Kullanıcı "Mont önerir misin?" dediğinde, asistan instructions.csv'deki genel prompt ile kategori bilgisini anlar ve Qdrant'tan mont kategorisindeki ürünlerden birini önerir. Kullanıcı beğenmediğinde, yine aynı kategoriden başka bir ürün önerilir. 