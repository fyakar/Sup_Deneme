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

# Tavsiye Fonksiyonu (kategori filtresi dahil)
def item_based_recommendation(product_name, selected_category=None, top_n=5):
    if product_name not in product_names:
        return [], [], []
    
    # Ürün adından index buluyoruz
    product_idx = product_names.index(product_name)
    
    # Ürünler arası benzerlikleri alıyoruz
    similarity_scores = cosine_sim[product_idx]
    
    # En yüksek benzerlik skorlarına sahip ürünlerin indekslerini alıyoruz
    recommended_product_idx = np.argsort(similarity_scores)[::-1]
    
    # Kendisi dahil edilmesin
    recommended_product_idx = recommended_product_idx[recommended_product_idx != product_idx]
    
    # Ürün adlarını ve skorları çekelim
    recommended_products = [product_names[idx] for idx in recommended_product_idx]
    recommendation_scores = similarity_scores[recommended_product_idx]
    
    # Eğer kategori filtresi seçilmişse, filtre uygula
    if selected_category:
        filtered = [(prod, score) for prod, score in zip(recommended_products, recommendation_scores) 
                    if product_category_map.get(prod) == selected_category]
        if not filtered:
            return [], [], []
        recommended_products, recommendation_scores = zip(*filtered)
    
    # İlk top_n ürün
    recommended_products = list(recommended_products)[:top_n]
    recommendation_scores = np.array(recommendation_scores)[:top_n]
    
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

    # Kategori filtresi seçimi
    all_categories = sorted(df['Category'].dropna().unique().tolist())
    selected_category = st.selectbox('Kategori filtresi uygula (İsteğe bağlı):', ['Tüm Kategoriler'] + all_categories)

    # Tavsiye butonu
    if st.button("Tavsiyeleri Göster"):
        # Eğer 'Tüm Kategoriler' seçildiyse kategori filtresi uygulama
        category_filter = None if selected_category == 'Tüm Kategoriler' else selected_category
        
        recommendations, recommendation_scores, recommendation_categories = item_based_recommendation(
            product_name_input, 
            selected_category=category_filter,
            top_n=top_n_input
        )
        
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
            st.warning("Bu ürün için seçilen kategoride tavsiye bulunamadı.")
