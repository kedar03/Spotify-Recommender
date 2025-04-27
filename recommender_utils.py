import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from skimage import io

# --- One-Hot Encoding Helper ---
def ohe_prep(df, column, new_column):
    tf_df = pd.get_dummies(df[column])
    feature_columns = tf_df.columns
    tf_df.columns = [new_column + "|" + str(i) for i in feature_columns]
    tf_df.reset_index(drop=True, inplace=True)
    return tf_df

# --- Feature Engineering ---
def create_feature_set(df, float_cols):
    tfidf = TfidfVectorizer(max_features=2000)
    tfidf_matrix = tfidf.fit_transform(df['genres_consolidated'].apply(lambda x: " ".join(x)))
    genre_df = pd.DataFrame(tfidf_matrix.toarray())
    genre_df.columns = ['genre|' + i for i in tfidf.get_feature_names_out()]
    genre_df.reset_index(drop=True, inplace=True)

    year_ohe = ohe_prep(df, 'year', 'year') * 0.5
    popularity_ohe = ohe_prep(df, 'popularity_bucket', 'popularity_new') * 0.15

    floats = df[float_cols].reset_index(drop=True)
    scaler = MinMaxScaler()
    floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns=floats.columns) * 0.2

    final = pd.concat([genre_df, floats_scaled, popularity_ohe, year_ohe], axis=1)
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
def generate_playlist_recos_2(df, features, nonplaylist_features, sp, fallback_url='https://i.ibb.co/rx5DFbs/spotify-logo.png'):
    non_playlist_df = df[df['id'].isin(nonplaylist_features['id'].values)].copy()
    non_playlist_df['sim'] = cosine_similarity(
        nonplaylist_features.drop('id', axis=1).values,
        features.values.reshape(1, -1)
    )[:, 0]

    non_playlist_df_top_40 = non_playlist_df.sort_values('sim', ascending=False).head(40).copy()

    track_ids = non_playlist_df_top_40['id'].tolist()
    tracks_info = sp.tracks(track_ids)['tracks']

    track_id_to_url = {}
    for track in tracks_info:
        images = track['album']['images']
        url = images[1]['url'] if len(images) > 1 else images[0]['url'] if images else fallback_url
        track_id_to_url[track['id']] = url

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
        plt.xlabel(df['name'].values[i], fontsize=12)
        plt.tight_layout(h_pad=0.4, w_pad=0)
        plt.subplots_adjust(wspace=None, hspace=None)

    plt.show()
