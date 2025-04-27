import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import matplotlib.pyplot as plt

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

# Ürün isimleri ve kategorileri eşleştir
product_category_map = df[['Product_Name', 'Category']].drop_duplicates().set_index('Product_Name')['Category'].to_dict()

# Tavsiye Fonksiyonu
def item_based_recommendation(product_name, top_n=5):
    if product_name not in product_names:
        return [], [], []
    
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
    
    # Ürünlerin kategorilerini çekelim
    recommended_categories = [product_category_map.get(prod, "Kategori Bilinmiyor") for prod in recommended_products]
    
    return recommended_products, recommendation_percentages, recommended_categories

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
        recommendations, recommendation_scores, recommendation_categories = item_based_recommendation(product_name_input, top_n=top_n_input)
        
        if recommendations:
            st.success(f"**{product_name_input}** ürününü alanlar şunları da alabilir:")

            # DataFrame halinde tablo gösterelim
            recommendation_df = pd.DataFrame({
                'Ürün Adı': recommendations,
                'Kategori': recommendation_categories,
                'Tavsiye Oranı (%)': recommendation_scores.round(2)
            })
            
            st.dataframe(recommendation_df)

            # Bar chart ile görsel gösterelim
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(recommendation_df['Ürün Adı'], recommendation_df['Tavsiye Oranı (%)'])
            ax.set_xlabel('Tavsiye Oranı (%)')
            ax.set_ylabel('Ürünler')
            ax.set_title('Tavsiye Edilen Ürünler ve Tavsiye Oranları')
            ax.invert_yaxis()  # En yüksek oran yukarıda olsun
            st.pyplot(fig)
            
        else:
            st.warning("Bu ürün için yeterli veri bulunamadı.")
