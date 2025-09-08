import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import joblib
import datetime

# Judul
st.title("Analisis Sentimen Twitter terhadap Revisi UU TNI")

# Load Model & Data 
try:
    model = joblib.load('model/model_naive_bayes_sentimen.pkl') 
    tfidf = joblib.load('model/tfidf_vectorizer.pkl') 

    
except FileNotFoundError as e:
    st.error(f"Error: File model atau TF-IDF tidak ditemukan. Pastikan path benar. Detail: {e}")
    st.stop()

try:
    df = pd.read_csv("data/data_analisis_final.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    
except FileNotFoundError as e:
    st.error(f"Error: File dataset '{e.filename}' tidak ditemukan. Pastikan path benar.")
    st.stop()
except Exception as e:
    st.error(f"Error saat memuat atau memproses dataset: {e}. Pastikan kolom 'tanggal', 'sentimen', 'text_clean', dan 'text' ada di file CSV.")
    st.stop()

# --- Filter Interaktif ---
st.sidebar.header("Filter Data")

# Ambil rentang tanggal dari dataframe
min_date_df = df['Date'].min().date() # sudah objek datetime.date
max_date_df = df['Date'].max().date() # sudah objek datetime.date

start_date = st.sidebar.date_input("Tanggal Mulai", min_date_df)
end_date = st.sidebar.date_input("Tanggal Selesai", max_date_df)

# Ambil opsi sentimen unik dari data
if 'sentimen' in df.columns:
    all_sentiments = df['sentimen'].unique().tolist()
    selected_sentimen = st.sidebar.multiselect("Pilih Sentimen", options=all_sentiments, default=all_sentiments)
else:
    st.error("Kolom 'sentimen' tidak ditemukan di dataset Anda. Filter sentimen tidak tersedia.")
    st.stop()

# --- Filter Data ---
filtered_df = df[
    (df['Date'].dt.date >= start_date) &
    (df['Date'].dt.date <= end_date) &
    (df['sentimen'].isin(selected_sentimen))
]

# --- Tampilkan Konten Utama ---
if not filtered_df.empty:
    # Ringkasan Sentimen
    st.subheader("Distribusi Sentimen")
    sentiment_count = filtered_df['sentimen'].value_counts()
    st.bar_chart(sentiment_count)

    # Tren Sentimen
    st.subheader("Tren Sentimen dari Waktu ke Waktu")
    trend = filtered_df.groupby([filtered_df['Date'].dt.date, 'sentimen']).size().unstack(fill_value=0)
    st.line_chart(trend)

    # Word Cloud
    st.subheader("Word Cloud per Sentimen")
    sentimen_pilihan_wc = st.selectbox("Pilih Sentimen untuk Word Cloud", options=all_sentiments)

    if 'text_clean' in filtered_df.columns:
        text_for_wc = " ".join(filtered_df[filtered_df['sentimen'] == sentimen_pilihan_wc]['text_clean'].astype(str))
        if text_for_wc:
            wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(text_for_wc)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info(f"Tidak ada teks untuk sentimen '{sentimen_pilihan_wc}' dalam rentang filter yang dipilih.")
    else:
        st.error("Kolom 'text_clean' tidak ditemukan di dataset Anda. Word Cloud tidak dapat dibuat.")

    # Contoh Tweet
    st.subheader("Contoh Tweet")
    if 'Text' in filtered_df.columns:
        st.dataframe(filtered_df[['Date', 'Text', 'sentimen']].sample(min(5, len(filtered_df))))
    else:
        st.error("Kolom 'text' tidak ditemukan di dataset Anda untuk menampilkan contoh tweet.")

else:
    st.warning("Tidak ada data yang cocok dengan filter yang dipilih. Silakan sesuaikan filter Anda.")

# Statistik Umum
st.subheader("Statistik Dataset Keseluruhan")
st.write(f"Total Tweet: {len(df)}")
st.write(f"Rentang Tanggal: {df['Date'].min().strftime('%d-%m-%Y')} s.d. {df['Date'].max().strftime('%d-%m-%Y')}")