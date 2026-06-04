import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astroquery.gaia import Gaia
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import hdbscan
import umap
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Stellar Clustering", page_icon="🌟", layout="wide")

# Title
st.title("🌟 Stellar Population Clustering")
st.markdown("**Live analysis of real stars from NASA/ESA Gaia DR3 catalog**")

# Sidebar controls
st.sidebar.header("⚙️ Controls")
n_stars = st.sidebar.slider("Number of stars to fetch", 500, 3000, 1000, 500)
algorithm = st.sidebar.selectbox("Clustering Algorithm", ["K-Means", "HDBSCAN"])
n_clusters = st.sidebar.slider("Number of clusters (K-Means)", 2, 10, 5)

# Fetch data
@st.cache_data
def fetch_data(n_stars):
    query = f"""
    SELECT TOP {n_stars}
        source_id, parallax, pmra, pmdec, phot_g_mean_mag, bp_rp
    FROM gaiadr3.gaia_source
    WHERE parallax > 0 AND ruwe < 1.4
    """
    st.info("Fetching live star data from NASA/ESA Gaia DR3...")
    job = Gaia.launch_job(query)
    results = job.get_results()
    df = results.to_pandas()
    return df

# Load and process
df = fetch_data(n_stars)
df = df.dropna().drop(columns=['source_id'])

# Scale
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Stats row
col1, col2, col3 = st.columns(3)
col1.metric("⭐ Total Stars", len(df))
col2.metric("📊 Features", df.shape[1])
col3.metric("🔭 Data Source", "Gaia DR3")

st.divider()

# Clustering
if algorithm == "K-Means":
    st.subheader(f"🔵 K-Means Clustering (K={n_clusters})")
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(df_scaled)
    n_noise = 0
else:
    st.subheader("🟣 HDBSCAN Clustering")
    model = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=3)
    labels = model.fit_predict(df_scaled)
    n_noise = (labels == -1).sum()

df['cluster'] = labels

# Metrics
unique_clusters = len(set(labels)) - (1 if -1 in labels else 0)
col1, col2 = st.columns(2)
col1.metric("🔍 Clusters Found", unique_clusters)
if algorithm == "HDBSCAN":
    col2.metric("⚫ Noise Stars", n_noise)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Parallax vs Brightness
scatter = axes[0].scatter(
    df['parallax'], df['phot_g_mean_mag'],
    c=labels, cmap='tab10', alpha=0.5, s=5
)
axes[0].set_xlabel('Parallax (distance from Earth)')
axes[0].set_ylabel('Brightness (G magnitude)')
axes[0].set_title('Parallax vs Brightness')
axes[0].invert_yaxis()
plt.colorbar(scatter, ax=axes[0])

# UMAP
st.info("Computing UMAP projection...")
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(df_scaled)

axes[1].scatter(
    embedding[:, 0], embedding[:, 1],
    c=labels, cmap='tab10', alpha=0.5, s=5
)
axes[1].set_xlabel('UMAP Dimension 1')
axes[1].set_ylabel('UMAP Dimension 2')
axes[1].set_title('UMAP — All 5 Features')

st.pyplot(fig)

st.divider()

# Cluster table
st.subheader("📋 Cluster Summary")
summary = df.groupby('cluster').agg({
    'parallax': 'mean',
    'phot_g_mean_mag': 'mean',
    'bp_rp': 'mean',
    'pmra': 'mean'
}).round(3)
summary.index.name = 'Cluster (-1 = noise)'
st.dataframe(summary)

# Raw data
if st.checkbox("Show raw star data"):
    st.dataframe(df)

st.markdown("---")
st.markdown("Built with ❤️ using NASA/ESA Gaia DR3 | K-Means | HDBSCAN | UMAP")