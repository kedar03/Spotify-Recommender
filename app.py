import streamlit as st
import pandas as pd
from recommender_utils import create_feature_set, generate_playlist_recos, visualize_songs

st.set_page_config(layout="wide")
st.title("🎵 Spotify Playlist Recommender (Manual Mode)")

st.markdown("""
Upload a playlist you've created manually (CSV file) or use our sample playlist.
We'll generate Top 40 recommendations based on your playlist's vibe using cosine similarity.
""")

# Load full track feature database
@st.cache_data
def load_tracks_final():
    return pd.read_csv('tracks_final.csv')

# Load sample playlist
@st.cache_data
def load_sample_playlist():
    return pd.read_csv('playlist_mumbai.csv')

tracks_final_df = load_tracks_final()
sample_playlist_df = load_sample_playlist()

# Upload CSV
uploaded_file = st.file_uploader("Upload your playlist CSV", type="csv")

if uploaded_file is not None:
    playlist_df = pd.read_csv(uploaded_file)
    st.success("Custom playlist uploaded!")
else:
    playlist_df = sample_playlist_df
    st.info("No playlist uploaded. Using sample playlist.")

# Show preview of playlist data
st.subheader("📄 Playlist Preview")
st.dataframe(playlist_df.head())

required_cols = {'id'}
if not required_cols.issubset(set(playlist_df.columns)):
    st.error("CSV must contain at least an 'id' column with Spotify track IDs.")
else:
    # Assume float columns are all columns except id, name, url
    float_cols = [col for col in tracks_final_df.columns if col not in ['id', 'name', 'url']]

    if st.button("🎯 Generate Recommendations"):
        st.info("Calculating recommendations...")

        # Prepare feature sets
        full_feature_set = create_feature_set(tracks_final_df, float_cols)
        playlist_features = full_feature_set[full_feature_set['id'].isin(playlist_df['id'])]
        playlist_vector = playlist_features.drop(columns=['id']).mean()

        nonplaylist_features = full_feature_set[~full_feature_set['id'].isin(playlist_df['id'])]

        # Generate recommendations
        recs = generate_playlist_recos(full_feature_set, playlist_vector, nonplaylist_features)
        st.success("Done! Showing top 40 recommendations:")

        for i, row in recs.iterrows():
            st.image(row['url'], width=100, caption=row['id'])
