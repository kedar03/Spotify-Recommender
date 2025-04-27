import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from skimage import io

# --- One-Hot Encoding Helper ---
def ohe_prep(df, column, new_column):
    if column not in df.columns:
        df[column] = "Unknown"
    tf_df = pd.get_dummies(df[column])
    feature_columns = tf_df.columns
    tf_df.columns = [new_column + "|" + str(i) for i in feature_columns]
    tf_df.reset_index(drop=True, inplace=True)
    return tf_df

# --- Feature Engineering ---
def create_feature_set(df, float_cols):
    tfidf = TfidfVectorizer(max_features=2000)

    # Ensure genres_consolidated exists
    if 'genres_consolidated' not in df.columns:
        df['genres_consolidated'] = ''

    df['genres_consolidated'] = df['genres_consolidated'].fillna('').astype(str)

    # Safely parse stringified lists
    def safe_parse(x):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return " ".join(parsed)
            else:
                return str(parsed)
        except Exception:
            return str(x)

    df['genres_consolidated'] = df['genres_consolidated'].apply(safe_parse)

    if df['genres_consolidated'].str.strip().replace('', np.nan).dropna().empty:
        genre_df = pd.DataFrame()
    else:
        tfidf_matrix = tfidf.fit_transform(df['genres_consolidated'])
        genre_df = pd.DataFrame(tfidf_matrix.toarray())
        genre_df.columns = ['genre|' + i for i in tfidf.get_feature_names_out()]
        genre_df.reset_index(drop=True, inplace=True)

    year_ohe = ohe_prep(df, 'year', 'year') * 0.5
    popularity_ohe = ohe_prep(df, 'popularity_bucket', 'popularity_new') * 0.15

    floats = df[float_cols].select_dtypes(include=[np.number]).reset_index(drop=True)
