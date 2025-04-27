import streamlit as st
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from recommender_utils import create_feature_set, generate_playlist_feature, generate_playlist_recos_2, visualize_songs

# Streamlit App Setup
st.set_page_config(layout="wide")
st.title("🎵 Spotify Playlist Recommender (Manual Mode)")

st.markdown("""
Upload your own playlist (CSV with track IDs and date_added), or use the sample playlist.
We will recommend 40 songs based on your playlist's vibe!
""")

# Spotify API Credentials
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=st.secrets["SPOTIPY_CLIENT_ID"],
    client_secret=st.secrets["SPOTIPY_CLIENT_SECRET"]
))

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

required_cols = {'id', 'date_added'}
if not required_cols.issubset(set(playlist_df.columns)):
    st.error("CSV must contain at least 'id' and 'date_added' columns.")
else:
    float_cols = [col for col in tracks_final_df.columns if col not in ['id', 'name', 'url', 'genres_consolidated', 'year', 'popularity_bucket']]

    if st.button("🎯 Generate Recommendations"):
        st.info("Calculating recommendations...")

        # Prepare feature sets
        full_feature_set = create_feature_set(tracks_final_df, float_cols)

        playlist_vector, nonplaylist_features = generate_playlist_feature(full_feature_set, playlist_df, weight_factor=1.09)

        # Generate recommendations
        recs = generate_playlist_recos_2(full_feature_set, playlist_vector, nonplaylist_features, sp=sp)
        st.success("Done! Showing top 40 recommendations:")

        visualize_songs(recs, fallback_image="https://i.ibb.co/rx5DFbs/spotify-logo.png")
