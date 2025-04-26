import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
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
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(max_features=500)
    tfidf_matrix = tfidf.fit_transform(df['genres_consolidated'].apply(lambda x: " ".join(x)))
    genre_df = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix)
    genre_df.columns = ['genre|' + i for i in tfidf.get_feature_names_out()]
    genre_df.reset_index(drop=True, inplace=True)

    year_ohe = ohe_prep(df, 'year', 'year') * 0.5
    popularity_ohe = ohe_prep(df, 'popularity_bucket', 'popularity') * 0.5

    floats = df[float_cols]
    scaler = MinMaxScaler()
    floats_scaled = pd.DataFrame(scaler.fit_transform(floats).astype('float32'), columns=floats.columns) * 0.2

    final = pd.concat([genre_df, year_ohe, popularity_ohe, floats_scaled], axis=1, copy=False)
    final['id'] = df['id'].values
    return final

# --- Recommendation Engine ---
def generate_playlist_recos(df, features, nonplaylist_features, fallback_url='https://i.ibb.co/rx5DFbs/spotify-logo.png'):
    non_playlist_df = df[df['id'].isin(nonplaylist_features['id'].values)].copy()
    non_playlist_df['sim'] = cosine_similarity(
        nonplaylist_features.drop('id', axis=1).values,
        features.values.reshape(1, -1)
    )[:, 0]

    non_playlist_df_top_40 = non_playlist_df.sort_values('sim', ascending=False).head(40).copy()

    # Use fallback Spotify logo
    non_playlist_df_top_40['url'] = fallback_url
    return non_playlist_df_top_40

# --- Visualization ---
def visualize_songs(df, default_image_url='https://i.ibb.co/rx5DFbs/spotify-logo.png'):
    import matplotlib.pyplot as plt

    temp = df['url'].values
    plt.figure(figsize=(15, int(0.625 * len(temp))))
    columns = 5

    for i, url in enumerate(temp):
        plt.subplot(len(temp) // columns + 1, columns, i + 1)
        try:
            if pd.isna(url) or url == '':
                raise ValueError("Missing URL")
            image = io.imread(url)
        except:
            image = io.imread(default_image_url)

        plt.imshow(image)
        plt.xticks(color='w', fontsize=0.1)
        plt.yticks(color='w', fontsize=0.1)
        plt.xlabel(df['name'].values[i], fontsize=12)
        plt.tight_layout(h_pad=0.4, w_pad=0)
        plt.subplots_adjust(wspace=None, hspace=None)

    plt.show()
