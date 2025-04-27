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

# Gerekli kolonları filtrele
df = df[['Customer_ID', 'Product_Name', 'Sales', 'Category', 'Order_Date', 'Type']]

# Ürün İsimleri ve Kategorileri Eşle
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

# RFM Analizi Fonksiyonu
def rfm_segmentation(df):
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    reference_date = df['Order_Date'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('Customer_ID').agg({
        'Order_Date': lambda x: (reference_date - x.max()).days,
        'Customer_ID': 'count',
        'Sales': 'sum'
    }).rename(columns={'Order_Date': 'Recency', 'Customer_ID': 'Frequency', 'Sales': 'Monetary'})
    
    # RFM skorları
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1]).astype(int)
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5]).astype(int)
    
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

    # Segment atama
    def segment(x):
        if x['R_Score'] >= 4 and x['F_Score'] >= 4 and x['M_Score'] >= 4:
            return 'Champions'
        elif x['F_Score'] >= 4:
            return 'Loyal Customers'
        elif x['R_Score'] <= 2 and x['F_Score'] <= 2:
            return 'At Risk'
        elif x['R_Score'] >= 4:
            return 'New Customers'
        else:
            return 'Others'
    
    rfm['Segment'] = rfm.apply(segment, axis=1)
    return rfm

# Streamlit Başlangıç
st.set_page_config(page_title="Superstore Dashboard", layout="wide")
st.title("📈 Superstore Veri Analizi ve Ürün Tavsiye Dashboardu")

# Sidebar
st.sidebar.image('logo.png', use_container_width=True)
st.sidebar.write("Datamigos - Data Analytics Team")
tabs = st.sidebar.radio('Menü Seçin:', ['Ürün Tavsiyesi', 'Genel Satış Analizi', 'Müşteri Segment Analizi'])

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
                'No': list(range(1, len(recommendations)+1)),
                'Ürün Adı': recommendations,
                'Kategori': recommendation_categories,
                'Tavsiye Oranı (%)': recommendation_scores.round(2)
            })
            st.dataframe(recommendation_df)

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

    if 'Type' in df.columns:
        type_options = ['Tüm Tipler'] + sorted(df['Type'].dropna().unique().tolist())
        selected_type = st.selectbox('Müşteri Tipi (Opsiyonel):', type_options)
    else:
        selected_type = 'Tüm Tipler'

    if selected_type != 'Tüm Tipler':
        filtered_df = df[df['Type'] == selected_type]
    else:
        filtered_df = df.copy()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Toplam Satış", f"${filtered_df['Sales'].sum():,.2f}")
    with col2:
        st.metric("Toplam Müşteri", filtered_df['Customer_ID'].nunique())
    with col3:
        st.metric("Toplam Ürün", filtered_df['Product_Name'].nunique())

    st.subheader('Kategorilere Göre Satışlar')
    category_sales = filtered_df.groupby('Category')['Sales'].sum().sort_values(ascending=False)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    category_sales.plot(kind='bar', ax=ax2, color='skyblue', edgecolor='black')
    ax2.set_ylabel('Toplam Satış ($)')
    ax2.set_title('Kategori Bazında Satışlar')
    plt.tight_layout()
    st.pyplot(fig2)

# Müşteri Segment Analizi Sayfası
elif tabs == 'Müşteri Segment Analizi':
    st.header('🧩 Müşteri Segment Analizi')

    rfm = rfm_segmentation(df)

    st.dataframe(rfm[['Recency', 'Frequency', 'Monetary', 'Segment']])

    st.subheader('Segment Dağılımı')
    segment_counts = rfm['Segment'].value_counts()

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    segment_counts.plot(kind='bar', ax=ax3, color='orange', edgecolor='black')
    ax3.set_ylabel('Müşteri Sayısı')
    ax3.set_title('Müşteri Segment Dağılımı')
    plt.tight_layout()
    st.pyplot(fig3)
