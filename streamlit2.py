import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Veriyi yükleme
file_path = 'CHURN HESAPLAMA.xlsx'  # Excel dosyasının yolu
df = pd.read_excel(file_path)

# Eğer 'Type' kolonu yoksa, Segment'ten türetelim
if 'Type' not in df.columns:
    if 'Segment' in df.columns:
        df['Type'] = df['Segment'].apply(lambda x: 'B2B' if x in ['Corporate', 'Home Office'] else 'B2C')

# Öneri sistemi için gerekli sütunları seçiyoruz
df = df[['Customer_ID', 'Product_Name', 'Sales', 'Category', 'Type']]

# Kullanıcı-Ürün Etkileşim Matrisi oluşturuyoruz
interaction_matrix = df.pivot_table(index='Customer_ID', columns='Product_Name', values='Sales', aggfunc='sum', fill_value=0)

# Cosine Similarity hesaplama
cosine_sim = cosine_similarity(interaction_matrix.T)
product_names = interaction_matrix.columns.tolist()

# Ürün isimleri ve kategorileri eşleştiriyoruz
product_category_map = df[['Product_Name', 'Category']].drop_duplicates().set_index('Product_Name')['Category'].to_dict()

# Tavsiye Fonksiyonu (kategori filtresi dahil)
def item_based_recommendation(product_name, selected_category=None, top_n=5):
    if product_name not in product_names:
        return [], [], []
    
    product_idx = product_names.index(product_name)
    similarity_scores = cosine_sim[product_idx]
    
    recommended_product_idx = np.argsort(similarity_scores)[::-1]
    recommended_product_idx = recommended_product_idx[recommended_product_idx != product_idx]
    
    recommended_products = [product_names[idx] for idx in recommended_product_idx]
    recommendation_scores = similarity_scores[recommended_product_idx]
    
    if selected_category:
        filtered = [(prod, score) for prod, score in zip(recommended_products, recommendation_scores) 
                    if product_category_map.get(prod) == selected_category]
        if not filtered:
            return [], [], []
        recommended_products, recommendation_scores = zip(*filtered)
    
    recommended_products = list(recommended_products)[:top_n]
    recommendation_scores = np.array(recommendation_scores)[:top_n]
    
    max_similarity = recommendation_scores.max() if recommendation_scores.max() != 0 else 1
    recommendation_percentages = (recommendation_scores / max_similarity) * 100
    
    recommended_categories = [product_category_map.get(prod, "Kategori Bilinmiyor") for prod in recommended_products]
    
    return recommended_products, recommendation_percentages, recommended_categories

# Streamlit Başlangıç
st.title("Superstore Ürün Tavsiye ve Satış Analizi Sistemi")

# Sidebar
st.sidebar.image('logo.png', use_container_width=True)
st.sidebar.write("Datamigos Ürün Tavsiye ve Analiz Sistemi - Explore products you may like!")

# Sidebar - Sekmeler
tabs = st.sidebar.radio('Sekmeler:', ['Ürün Tavsiyesi', 'Genel Satış Analizi'])

if tabs == 'Ürün Tavsiyesi':
    st.subheader('Ürün Tavsiyesi')
    
    product_list = sorted(df['Product_Name'].unique().tolist())
    product_name_input = st.selectbox('Bir ürün seçin:', product_list)
    top_n_input = st.slider("Önerilecek ürün sayısını seçin", 1, 10, 5)

    all_categories = sorted(df['Category'].dropna().unique().tolist())
    selected_category = st.selectbox('Kategori filtresi uygula (İsteğe bağlı):', ['Tüm Kategoriler'] + all_categories)

    if st.button("Tavsiyeleri Göster"):
        category_filter = None if selected_category == 'Tüm Kategoriler' else selected_category
        recommendations, recommendation_scores, recommendation_categories = item_based_recommendation(
            product_name_input, 
            selected_category=category_filter,
            top_n=top_n_input
        )
        
        if recommendations:
            st.success(f"**{product_name_input}** ürününü alanlar şunları da alabilir:")

            # Tavsiye edilen ürünleri tablo olarak gösterelim
            recommendation_df = pd.DataFrame({
                'No': list(range(1, len(recommendations)+1)),  # 1'den başlatarak numaralandır
                'Ürün Adı': recommendations,
                'Kategori': recommendation_categories,
                'Tavsiye Oranı (%)': recommendation_scores.round(2)
            })
            
            st.dataframe(recommendation_df)

            # Renk skalası oluştur (yüksek oran koyu renk olacak)
            norm = plt.Normalize(recommendation_df['Tavsiye Oranı (%)'].min(), recommendation_df['Tavsiye Oranı (%)'].max())
            colors = plt.cm.Blues(norm(recommendation_df['Tavsiye Oranı (%)']))

            # Geliştirilmiş bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(recommendation_df['Ürün Adı'], recommendation_df['Tavsiye Oranı (%)'], 
                           color=colors, edgecolor='black')

            # Barların üstüne değerleri yazalım
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center', fontsize=10, color='black')

            # Eksen etiketleri ve başlık
            ax.set_xlabel('Tavsiye Oranı (%)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Ürünler', fontsize=12, fontweight='bold')
            ax.set_title('Tavsiye Edilen Ürünler ve Tavsiye Oranları', fontsize=14, fontweight='bold', pad=20)

            # Üst ve sağ çizgileri kaldır
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Y ekseni ters çevir (en yüksek oran üstte olsun)
            ax.invert_yaxis()

            plt.tight_layout()
            st.pyplot(fig)
            
        else:
            st.warning("Bu ürün için seçilen kategoride tavsiye bulunamadı.")

elif tabs == 'Genel Satış Analizi':
    st.subheader('Genel Satış Analizi')

    # Type filtresi ekleyelim (isteğe bağlı)
    if 'Type' in df.columns:
        type_options = ['Tüm Tipler'] + sorted(df['Type'].dropna().unique().tolist())
        selected_type = st.selectbox('Müşteri Tipi Seçin (Opsiyonel):', type_options)
    else:
        selected_type = 'Tüm Tipler'

    # Type filtresine göre veriyi filtreleyelim
    if selected_type != 'Tüm Tipler':
        filtered_df = df[df['Type'] == selected_type]
    else:
        filtered_df = df.copy()

    # Genel metrikler
    total_sales = filtered_df['Sales'].sum()
    total_customers = filtered_df['Customer_ID'].nunique()
    total_products = filtered_df['Product_Name'].nunique()

    st.metric("Toplam Satış", f"${total_sales:,.2f}")
    st.metric("Toplam Müşteri", total_customers)

    # En çok satan kategoriler grafiği
    category_sales = filtered_df.groupby('Category')['Sales'].sum().sort_values(ascending=False)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    category_sales.plot(kind='bar', ax=ax2, color='skyblue', edgecolor='black')
    ax2.set_ylabel('Toplam Satış ($)')
    ax2.set_title('Kategorilere Göre Satışlar')
    plt.tight_layout()
    st.pyplot(fig2)
