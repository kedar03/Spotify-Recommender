import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from skimage import io

# --- One-Hot Encoding Helper ---
def ohe_prep(df, column, new_column):
    if column not in df.columns:
        df[column] = "Unknown"  # Safe default if column missing
    tf_df = pd.get_dummies(df[column])
    feature_columns = tf_df.columns
    tf_df.columns = [new_column + "|" + str(i) for i in feature_columns]
    tf_df.reset_index(drop=True, inplace=True)
    return tf_df

# --- Feature Engineering ---
def create_feature_set(df, float_cols):
    tfidf = TfidfVectorizer(max_features=2000)

    # Ensure genres_consolidated column exists
    if 'genres_consolidated' not in df.columns:
        df['genres_consolidated'] = ''

    # Fill NaNs and ensure string type
    df['genres_consolidated'] = df['genres_consolidated'].fillna('').astype(str)

    # TF-IDF on cleaned genre text
    tfidf_matrix = tfidf.fit_transform(df['genres_consolidated'])
    genre_df = pd.DataFrame(tfidf_matrix.toarray())
    genre_df.columns = ['genre|' + i for i in tfidf.get_feature_names_out()]
    genre_df.reset_index(drop=True, inplace=True)

    # OHE year and popularity
    year_ohe = ohe_prep(df, 'year', 'year') * 0.5
    popularity_ohe = ohe_prep(df, 'popularity_bucket', 'popularity_new') * 0.15

    # Scale float columns safely
    floats = df[float_cols].select_dtypes(include=[np.number]).reset_index(drop=True)
    scaler = MinMaxScaler()
    floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns=floats.columns) * 0.2

    # Combine all features safely
    all_parts = [genre_df, floats_scaled, popularity_ohe, year_ohe]
    all_parts = [part.reset_index(drop=True) for part in all_parts if not part.empty]

    final = pd.concat(all_parts, axis=1)
    final['id'] = df['id'].values
    return final

# --- Playlist Feature Generation ---
def generate_playlist_feature(complete_feature_set, playlist_df, weight_factor):
    complete_feature_set_playlist = complete_feature_set[complete_feature_set['id'].isin(playlist_df['id'].values)]
    complete_feature_set_playlist = complete_feature_set_playlist.merge(playlist_df[['id', 'date_added']], on='id', how='inner')
    complete_feature_set_nonplaylist = complete_feature_set[~complete_feature_set['id'].isin(playlist_df['id'].values)]

    playlist_feature_set = complete_feature_set_playlist.sort_values('date_added', ascending=False)
    most_recent_date = playlist_feature_set.iloc[0, -1]

    for ix, row in playlist_feature_set.iterrows():
        playlist_feature_set.loc[ix, 'months_from_recent'] = int((most_recent_date.to_pydatetime() - row.iloc[-1].to_pydatetime()).days / 30)

    playlist_feature_set['weight'] = playlist_feature_set['months_from_recent'].apply(lambda x: weight_factor ** (-x))

    playlist_feature_set_weighted = playlist_feature_set.copy()
    playlist_feature_set_weighted.update(playlist_feature_set_weighted.iloc[:, :-4].mul(playlist_feature_set_weighted['weight'], 0))
    playlist_feature_set_weighted_final = playlist_feature_set_weighted.iloc[:, :-4]

    return playlist_feature_set_weighted_final.sum(axis=0), complete_feature_set_nonplaylist

# --- Recommendation Engine ---
def generate_playlist_recos_2(df, features, nonplaylist_features, sp=None, fallback_url='https://i.ibb.co/rx5DFbs/spotify-logo.png'):
    non_playlist_df = df[df['id'].isin(nonplaylist_features['id'].values)].copy()
    non_playlist_df['sim'] = cosine_similarity(
        nonplaylist_features.drop('id', axis=1).values,
        features.values.reshape(1, -1)
    )[:, 0]

    non_playlist_df_top_40 = non_playlist_df.sort_values('sim', ascending=False).head(40).copy()

    track_ids = non_playlist_df_top_40['id'].tolist()

    # Handle album artwork safely
    track_id_to_url = {}

    if sp is not None:
        try:
            tracks_info = sp.tracks(track_ids)['tracks']
            for track in tracks_info:
                images = track['album']['images']
                url = images[1]['url'] if len(images) > 1 else images[0]['url'] if images else fallback_url
                track_id_to_url[track['id']] = url
        except Exception:
            # If Spotify API fails, fallback for all
            track_id_to_url = {track_id: fallback_url for track_id in track_ids}
    else:
        track_id_to_url = {track_id: fallback_url for track_id in track_ids}

    non_playlist_df_top_40['url'] = non_playlist_df_top_40['id'].map(track_id_to_url)
    return non_playlist_df_top_40

# --- Visualization ---
def visualize_songs(df, fallback_image):
    temp = df['url'].values
    plt.figure(figsize=(15, int(0.625 * len(temp))))
    columns = 5

    for i, url in enumerate(temp):
        plt.subplot(len(temp) // columns + 1, columns, i + 1)

        try:
            if pd.isna(url) or url == '':
                raise ValueError("Invalid URL")
            image = io.imread(url)
        except:
            image = io.imread(fallback_image)

        plt.imshow(image)
        plt.xticks(color='w', fontsize=0.1)
        plt.yticks(color='w', fontsize=0.1)
        plt.xlabel(df['name'].values[i] if 'name' in df.columns else '', fontsize=12)
        plt.tight_layout(h_pad=0.4, w_pad=0)
        plt.subplots_adjust(wspace=None, hspace=None)

    plt.show()
