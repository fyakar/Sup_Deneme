import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import matplotlib.pyplot as plt

# Veriyi yükleme
file_path = 'CHURN HESAPLAMA.xlsx'  # Excel dosyasının yolu
df = pd.read_excel(file_path)

# Eğer 'Type' kolonu yoksa Segment'ten türetelim
if 'Type' not in df.columns:
    if 'Segment' in df.columns:
        df['Type'] = df['Segment'].apply(lambda x: 'B2B' if x in ['Corporate', 'Home Office'] else 'B2C')

# Gerekli kolonları seç
df = df[['Customer_ID', 'Product_Name', 'Sales', 'Category', 'Order_Date', 'Type']]

# Order_Date kolonunu datetime yap
df['Order_Date'] = pd.to_datetime(df['Order_Date'])

# Ürün isimleri ve kategorileri eşle
product_category_map = df[['Product_Name', 'Category']].drop_duplicates().set_index('Product_Name')['Category'].to_dict()

# Kullanıcı-Ürün Etkileşim Matrisi
interaction_matrix = df.pivot_table(index='Customer_ID', columns='Product_Name', values='Sales', aggfunc='sum', fill_value=0)

# Cosine Similarity Hesaplama
cosine_sim = cosine_similarity(interaction_matrix.T)
product_names = interaction_matrix.columns.tolist()

# Tavsiye Fonksiyonu
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
st.set_page_config(page_title="Superstore Dashboard", layout="wide")
st.title("📈 Superstore Veri Analizi ve Ürün Tavsiye Dashboardu")

# Sidebar
st.sidebar.image('logo.png', use_container_width=True)
st.sidebar.write("Datamigos - Data Analytics Team")
tabs = st.sidebar.radio('Menü Seçin:', ['Ürün Tavsiyesi', 'Genel Satış Analizi'])

# Ürün Tavsiyesi Sayfası
if tabs == 'Ürün Tavsiyesi':
    st.header('🎯 Ürün Tavsiyesi')
    
    product_list = sorted(df['Product_Name'].unique().tolist())
    product_name_input = st.selectbox('Bir ürün seçin:', product_list)
    top_n_input = st.slider("Önerilecek ürün sayısı:", 1, 10, 5)

    all_categories = sorted(df['Category'].dropna().unique().tolist())
    selected_category = st.selectbox('Kategori filtresi (Opsiyonel):', ['Tüm Kategoriler'] + all_categories)

    if st.button("Tavsiyeleri Göster"):
        category_filter = None if selected_category == 'Tüm Kategoriler' else selected_category
        recommendations, recommendation_scores, recommendation_categories = item_based_recommendation(
            product_name_input, category_filter, top_n_input
        )
        
        if recommendations:
            recommendation_df = pd.DataFrame({
                'Sıra No': list(range(1, len(recommendations) + 1)),
                'Ürün Adı': recommendations,
                'Kategori': recommendation_categories,
                'Tavsiye Oranı (%)': recommendation_scores.round(2)
            })

            # Index gizleniyor
            st.dataframe(recommendation_df.reset_index(drop=True))

            norm = plt.Normalize(recommendation_df['Tavsiye Oranı (%)'].min(), recommendation_df['Tavsiye Oranı (%)'].max())
            colors = plt.cm.Blues(norm(recommendation_df['Tavsiye Oranı (%)']))

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(recommendation_df['Ürün Adı'], recommendation_df['Tavsiye Oranı (%)'], color=colors, edgecolor='black')
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center', fontsize=10, color='black')

            ax.set_xlabel('Tavsiye Oranı (%)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Ürünler', fontsize=12, fontweight='bold')
            ax.set_title('Tavsiye Edilen Ürünler ve Tavsiye Oranları', fontsize=14, fontweight='bold', pad=20)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Seçilen ürün ve kategori için tavsiye bulunamadı.")

# Genel Satış Analizi Sayfası
elif tabs == 'Genel Satış Analizi':
    st.header('💰 Genel Satış Analizi')

    # Type filtresi
    if 'Type' in df.columns:
        type_options = ['Tüm Tipler'] + sorted(df['Type'].dropna().unique().tolist())
        selected_type = st.selectbox('Müşteri Tipi (Opsiyonel):', type_options)
    else:
        selected_type = 'Tüm Tipler'

    # Tarih filtresi opsiyonel checkbox
    date_filter_active = st.checkbox("Tarih filtresi uygula", value=False)

    if date_filter_active:
        min_date = df['Order_Date'].min()
        max_date = df['Order_Date'].max()

        date_range = st.date_input(
            label="Tarih Aralığı Seçin",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )

        start_date, end_date = date_range
    else:
        # Eğer tarih filtresi uygulanmıyorsa tüm veriyi kullan
        start_date = df['Order_Date'].min()
        end_date = df['Order_Date'].max()

    # Filtrelemeler
    filtered_df = df.copy()

    if selected_type != 'Tüm Tipler':
        filtered_df = filtered_df[filtered_df['Type'] == selected_type]

    filtered_df = filtered_df[(filtered_df['Order_Date'] >= pd.to_datetime(start_date)) & (filtered_df['Order_Date'] <= pd.to_datetime(end_date))]

    # Metrikler
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Toplam Satış", f"${filtered_df['Sales'].sum():,.2f}")
    with col2:
        st.metric("Toplam Müşteri", filtered_df['Customer_ID'].nunique())

    st.subheader('Kategorilere Göre Satışlar')
    category_sales = filtered_df.groupby('Category')['Sales'].sum().sort_values(ascending=False)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    category_sales.plot(kind='bar', ax=ax2, color='skyblue', edgecolor='black')
    ax2.set_ylabel('Toplam Satış ($)')
    ax2.set_title('Kategori Bazında Satışlar')
    plt.tight_layout()
    st.pyplot(fig2)
