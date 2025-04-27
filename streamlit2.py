import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Veriyi yükleme
file_path = 'CHURN HESAPLAMA.xlsx'  # Excel dosyasının yolu
df = pd.read_excel(file_path)

# Öneri sistemi için gerekli sütunları seçiyoruz
df = df[['Customer_ID', 'Product_Name', 'Sales', 'Category']]

# Kullanıcı-Ürün Etkileşim Matrisi oluşturuyoruz (ÜRÜN İSMİ KULLANARAK)
interaction_matrix = df.pivot_table(index='Customer_ID', columns='Product_Name', values='Sales', aggfunc='sum', fill_value=0)

# Cosine Similarity ile ürünler arasındaki benzerlikleri hesaplıyoruz
cosine_sim = cosine_similarity(interaction_matrix.T)
product_names = interaction_matrix.columns.tolist()

# Tavsiye Fonksiyonu
def item_based_recommendation(product_name, top_n=5):
    if product_name not in product_names:
        return [], []
    
    # Ürün adından index buluyoruz
    product_idx = product_names.index(product_name)
    
    # Ürünler arası benzerlikleri alıyoruz
    similarity_scores = cosine_sim[product_idx]
    
    # En yüksek benzerlik skorlarına sahip N ürünü alıyoruz
    recommended_product_idx = np.argsort(similarity_scores)[::-1]
    
    # Kendisi dahil edilmesin
    recommended_product_idx = recommended_product_idx[recommended_product_idx != product_idx]
    
    # Tavsiye edilecek ürünler ve skorlar
    top_recommendations = recommended_product_idx[:top_n]
    recommended_products = [product_names[idx] for idx in top_recommendations]
    recommendation_scores = similarity_scores[top_recommendations]
    
    # Tavsiye oranı: (Benzerlik oranı / Maksimum benzerlik oranı) * 100
    max_similarity = recommendation_scores.max() if recommendation_scores.max() != 0 else 1
    recommendation_percentages = (recommendation_scores / max_similarity) * 100
    
    return recommended_products, recommendation_percentages

# Streamlit Başlangıç
st.title("Superstore Ürün Tavsiye Sistemi")

# Sidebar logo ve açıklama
st.sidebar.image('logo.png', use_container_width=True)
st.sidebar.write("Datamigos Ürün Tavsiye Sistemi - Explore products you may like!")

# Sidebar - Sekmeler
tabs = st.sidebar.radio('Sekmeler:', ['Ürün Tavsiyesi'])

# Sekme 1 - Ürün Tavsiyesi
if tabs == 'Ürün Tavsiyesi':
    st.subheader('Ürün Tavsiyesi')
    
    # Ürün listesini oluştur
    product_list = sorted(df['Product_Name'].unique().tolist())

    # Ürün seçimi
    product_name_input = st.selectbox('Bir ürün seçin:', product_list)
    
    # Önerilecek ürün sayısı
    top_n_input = st.slider("Önerilecek ürün sayısını seçin", 1, 10, 5)
    
    # Tavsiye butonu
    if st.button("Tavsiyeleri Göster"):
        recommendations, recommendation_scores = item_based_recommendation(product_name_input, top_n=top_n_input)
        
        if recommendations:
            st.write(f"**{product_name_input}** ürününü alanlar şunları da alabilir:")
            for i, (product, score) in enumerate(zip(recommendations, recommendation_scores), 1):
                st.write(f"{i}. {product} - Tavsiye Oranı: {score:.2f}%")
        else:
            st.warning("Bu ürün için yeterli veri bulunamadı.")
