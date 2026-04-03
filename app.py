import numpy as np
import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "indobenchmark/indobert-base-p1"
LABELS = ["positif", "netral", "negatif"]

# Kumpulan contoh kalimat untuk setiap kelas sebagai prototipe sentimen.
PROTOTYPE_TEXTS = {
    "positif": [
        "Saya sangat senang dengan pelayanan ini.",
        "Produknya bagus dan kualitasnya memuaskan.",
        "Pengalaman belanja saya luar biasa.",
        "Aplikasinya cepat dan mudah digunakan.",
    ],
    "netral": [
        "Pengiriman dilakukan pada hari Senin.",
        "Saya membeli produk ini minggu lalu.",
        "Harga produk ini sama seperti toko lain.",
        "Informasi pada halaman produk cukup jelas.",
    ],
    "negatif": [
        "Saya kecewa dengan layanan ini.",
        "Produknya rusak saat diterima.",
        "Aplikasi sering error dan lambat.",
        "Respon customer service sangat buruk.",
    ],
}


@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    pooled = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    denom = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return pooled / denom


def encode_texts(tokenizer, model, texts: list[str]) -> np.ndarray:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**encoded)

    embeddings = mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()


@st.cache_resource(show_spinner=True)
def build_prototype_embeddings():
    tokenizer, model = load_model_and_tokenizer()
    prototype_embeddings = {}

    for label, texts in PROTOTYPE_TEXTS.items():
        embs = encode_texts(tokenizer, model, texts)
        prototype_embeddings[label] = embs.mean(axis=0)

    return prototype_embeddings


def predict_sentiment(text: str):
    tokenizer, model = load_model_and_tokenizer()
    prototypes = build_prototype_embeddings()

    query_emb = encode_texts(tokenizer, model, [text])[0]

    scores = {}
    for label in LABELS:
        proto = prototypes[label]
        # Karena embedding sudah dinormalisasi, dot product = cosine similarity.
        scores[label] = float(np.dot(query_emb, proto))

    max_score = max(scores.values())
    exp_scores = {k: np.exp(v - max_score) for k, v in scores.items()}
    total = sum(exp_scores.values())
    probs = {k: v / total for k, v in exp_scores.items()}

    best_label = max(probs, key=probs.get)
    return best_label, probs


def main():
    st.set_page_config(page_title="Sentiment Analysis Bahasa Indonesia", page_icon="💬", layout="centered")

    st.title("Sentiment Analysis Bahasa Indonesia")
    st.caption("Model: indobenchmark/indobert-base-p1")

    st.info(
        "Aplikasi ini menggunakan IndoBERT base tanpa fine-tuning khusus sentimen. "
        "Prediksi dilakukan dengan membandingkan kemiripan embedding ke prototipe sentimen."
    )

    text = st.text_area(
        "Masukkan teks Indonesia:",
        placeholder="Contoh: Pelayanannya cepat dan sangat membantu.",
        height=160,
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        analyze = st.button("Analisis", use_container_width=True)

    if analyze:
        if not text.strip():
            st.warning("Teks tidak boleh kosong.")
            st.stop()

        with st.spinner("Memproses sentimen..."):
            label, probs = predict_sentiment(text.strip())

        label_map = {
            "positif": "😊 Positif",
            "netral": "😐 Netral",
            "negatif": "😞 Negatif",
        }

        st.subheader("Hasil Prediksi")
        st.success(f"Sentimen terdeteksi: **{label_map[label]}**")

        st.subheader("Skor Probabilitas")
        st.progress(float(probs["positif"]), text=f"Positif: {probs['positif']:.2%}")
        st.progress(float(probs["netral"]), text=f"Netral: {probs['netral']:.2%}")
        st.progress(float(probs["negatif"]), text=f"Negatif: {probs['negatif']:.2%}")

        st.json({k: round(v, 4) for k, v in probs.items()})


if __name__ == "__main__":
    main()
