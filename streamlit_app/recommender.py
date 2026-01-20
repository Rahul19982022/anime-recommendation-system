
import pandas as pd
import numpy as np
import pickle
import os
from sentence_transformers import util
from numpy.linalg import norm

# --- 1. LOAD ALL FILES ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJ_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    DATA_PATH = os.path.join(PROJ_ROOT, 'app_data')

    # Load DataFrames
    anime = pd.read_pickle(os.path.join(DATA_PATH, 'anime_with_posters.pkl'))
    anime_agg = pd.read_pickle(os.path.join(DATA_PATH, 'anime_agg_processed.pkl'))
    content_df = pd.read_pickle(os.path.join(PROJ_ROOT, 'datasets/created_datasets/content_df_model.pkl'))
    anime_genres_mlb = pd.read_pickle(os.path.join(DATA_PATH, 'anime_genres_mlb.pkl'))

    # Load IDs and Lists
    with open(os.path.join(DATA_PATH, 'divided_opinion_anime_ids.pkl'), 'rb') as f:
        divided_opinion_ids = pickle.load(f)
    with open(os.path.join(DATA_PATH, 'genres_list.pkl'), 'rb') as f:
        genres_list = pickle.load(f)

    # --- NEW: Load Relations Dictionary ---
    with open(os.path.join(DATA_PATH, 'anime_relations_expanded.pkl'), 'rb') as f:
        anime_relations = pickle.load(f)

    # Filter content_df
    content_df = content_df[content_df['MAL_ID'].isin(anime['MAL_ID'].unique())]

    # Load Weights
    with open(os.path.join(DATA_PATH, 'anime_model_weights.pkl'), 'rb') as f:
        anime_weights = pickle.load(f)

    # Load Encodings
    with open(os.path.join(PROJ_ROOT, 'datasets/created_datasets/encoded_dictionary/anime2anime_encoded.pkl'), 'rb') as f:
        anime2anime_encoded = pickle.load(f)
    with open(os.path.join(PROJ_ROOT, 'datasets/created_datasets/encoded_dictionary/anime_encoded2anime.pkl'), 'rb') as f:
        anime_encoded2anime = pickle.load(f)
    encoded_dictionary = {'anime2anime_encoded': anime2anime_encoded, 'anime_encoded2anime': anime_encoded2anime}

except Exception as e:
    print(f"Error loading files: {e}")
    anime = pd.DataFrame()
    anime_weights = None
    anime_relations = {}

# --- 2. CORE LOGIC FUNCTIONS ---
def get_anime_id_from_name(name, anime_df):
    try: return anime_df[anime_df['Name'].str.lower() == name.lower()]['MAL_ID'].values[0]
    except:
        try: return anime_df[anime_df['English name'].str.lower() == name.lower()]['MAL_ID'].values[0]
        except: return None

def get_anime_details(name, anime_df, anime_agg_df):
    anime_id = get_anime_id_from_name(name, anime_df)
    if anime_id is None: return None
    anime_info = anime_df[anime_df['MAL_ID'] == anime_id].copy()
    anime_info = anime_info.merge(anime_agg_df[['anime_id', 'anime_avg_rating']], left_on='MAL_ID', right_on='anime_id', how='left')
    if anime_info.empty: return None
    return anime_info

# --- 3. FILTER FUNCTION ---
def filter_recommendations(recommendations_df, anime_df, anime_agg_df, **filters):
    if recommendations_df is None or recommendations_df.empty:
        return recommendations_df

    if 'MAL_ID' not in recommendations_df.columns:
         recommendations_df.rename(columns={'anime_id': 'MAL_ID'}, inplace=True)

    needed_info_cols = ['Name', 'English name', 'Type', 'Genres_edited', 'Origin_year', 'Popularity_adjusted', 'image_url', 'synopsis']
    current_cols = set(recommendations_df.columns)
    missing_cols = set(needed_info_cols) - current_cols
    rec_with_info = recommendations_df.copy()

    if len(missing_cols) > 0:
        cols_to_merge = list(missing_cols) + ['MAL_ID']
        rec_with_info = rec_with_info.merge(anime_df[cols_to_merge], on='MAL_ID', how='left')

    rec_with_info = rec_with_info.merge(anime_agg_df[['anime_id', 'anime_avg_rating']], left_on='MAL_ID', right_on='anime_id', how='left')

    if filters.get('Type_preferred'):
        rec_with_info = rec_with_info[rec_with_info['Type'].isin(filters['Type_preferred'])]

    if filters.get('Genres_preferred'):
        genres_set = set(filters['Genres_preferred'])
        rec_with_info = rec_with_info[rec_with_info['Genres_edited'].fillna('').str.split('|').apply(lambda x: genres_set.issubset(x))]

    if filters.get('Origin_year_range'):
        year_range = filters['Origin_year_range']
        rec_with_info = rec_with_info[(rec_with_info['Origin_year'] >= year_range[0]) & (rec_with_info['Origin_year'] <= year_range[1])]

    if filters.get('min_anime_rating'):
        rec_with_info['anime_avg_rating'] = pd.to_numeric(rec_with_info['anime_avg_rating'], errors='coerce')
        rec_with_info = rec_with_info.dropna(subset=['anime_avg_rating'])
        rec_with_info = rec_with_info[rec_with_info['anime_avg_rating'] >= filters['min_anime_rating']]

    if filters.get('popularity_range'):
        pop_range = filters['popularity_range']
        rec_with_info = rec_with_info[
            (rec_with_info['Popularity_adjusted'] >= pop_range[0]) &
            (rec_with_info['Popularity_adjusted'] <= pop_range[1])
        ]

    return rec_with_info.reset_index(drop=True)


# --- 4. GETTER FUNCTIONS ---
def model_rec_based_on_anime_similarity(name, anime_df, weights, enc_dict):
    if weights is None: return None
    anime_id = get_anime_id_from_name(name, anime_df)
    if anime_id is None: return None
    encoded_index = enc_dict['anime2anime_encoded'].get(anime_id)
    if encoded_index is None: return None

    dists = np.dot(weights, weights[encoded_index])

    dists_df = pd.DataFrame(dists, columns=['similarity_model'])
    dists_df['MAL_ID'] = dists_df.index.map(enc_dict['anime_encoded2anime'])
    dists_df = dists_df[dists_df['MAL_ID'] != anime_id]
    return dists_df.sort_values(by='similarity_model', ascending=False)

def content_based_rec(name, anime_df, content_df):
    anime_id = get_anime_id_from_name(name, anime_df)
    if anime_id is None: return None
    try:
        user_anime_vector = content_df.loc[content_df['MAL_ID'] == anime_id, 'plot_embeddings'].values[0]
        scores = util.pytorch_cos_sim(user_anime_vector, np.array(content_df['plot_embeddings'].tolist()))[0]
        results = pd.DataFrame({'MAL_ID': content_df['MAL_ID'], 'similarity': scores.numpy()})
        results = results[results['MAL_ID'] != anime_id]
        return results.sort_values(by='similarity', ascending=False)
    except (IndexError, KeyError): return None

def rec_based_on_genre_similarity(name, anime_df, anime_genre_mlb_df):
    anime_id = get_anime_id_from_name(name, anime_df)
    if anime_id is None: return None
    try:
        if 'MAL_ID' not in anime_genre_mlb_df.columns: return None
        genre_columns = anime_genre_mlb_df.columns[2:]
        selected_anime_genre_vector = anime_genre_mlb_df.loc[anime_genre_mlb_df['MAL_ID'] == anime_id, genre_columns].values.reshape(-1, 1)
        anime_genre_array = anime_genre_mlb_df.loc[:, genre_columns].values

        similarity_scores = np.dot(anime_genre_array, selected_anime_genre_vector).reshape(-1)
        norm_factor = norm(anime_genre_array, axis=1) * norm(selected_anime_genre_vector)
        norm_factor = np.where(norm_factor == 0, 1e-6, norm_factor)
        similarity_scores = similarity_scores / norm_factor

        results = pd.DataFrame({'MAL_ID': anime_genre_mlb_df['MAL_ID'], 'similarity_genre': similarity_scores})
        results = results[results['MAL_ID'] != anime_id]
        return results.sort_values(by='similarity_genre', ascending=False)
    except (IndexError, KeyError): return None

def rec_based_on_comb_of_genre_sim_and_model(name, anime_df, weights, enc_dict, genre_mlb_df, threshold=0.5):
    if weights is None: return None
    rec_model = model_rec_based_on_anime_similarity(name, anime_df, weights, enc_dict)
    rec_genre = rec_based_on_genre_similarity(name, anime_df, genre_mlb_df)
    if rec_model is None or rec_genre is None: return None
    comb_rec = pd.merge(rec_genre, rec_model, on='MAL_ID', how='inner')
    comb_rec = comb_rec[comb_rec['similarity_genre'] >= threshold]
    return comb_rec.sort_values(by='similarity_model', ascending=False)

def get_divided_opinion_animes(anime_df, ids_list):
    if anime_df.empty: return pd.DataFrame()
    divided_df = anime_df[anime_df['MAL_ID'].isin(ids_list)]
    return divided_df

# --- 5. MAIN ORCHESTRATOR FUNCTIONS ---
def get_recommendations_by_name(name, rec_type, top_n=10, remove_related=False, **filters):
    if anime.empty or anime_weights is None: return pd.DataFrame()

    results_df = None
    if rec_type == 'Model-Based Similarity':
        results_df = model_rec_based_on_anime_similarity(name, anime, anime_weights, encoded_dictionary)
    elif rec_type == 'Content (Plot) Similarity':
        results_df = content_based_rec(name, anime, content_df)
    elif rec_type == 'Combined Model + Genre':
        results_df = rec_based_on_comb_of_genre_sim_and_model(name, anime, anime_weights, encoded_dictionary, anime_genres_mlb, threshold=filters.get('genre_threshold', 0.5))

    if results_df is None or results_df.empty: return None

    # --- NEW: Filter out related anime if requested ---
    if remove_related:
        input_id = get_anime_id_from_name(name, anime)
        # Check if we have relations data for this anime
        if input_id in anime_relations:
            related_ids = anime_relations[input_id]
            # Ensure proper column name for filtering
            if 'MAL_ID' not in results_df.columns and 'anime_id' in results_df.columns:
                 results_df.rename(columns={'anime_id': 'MAL_ID'}, inplace=True)

            # Remove any recommendation whose ID is in the related list
            if 'MAL_ID' in results_df.columns:
                results_df = results_df[~results_df['MAL_ID'].isin(related_ids)]
    # ------------------------------------------------

    filtered_results = filter_recommendations(results_df, anime, anime_agg, **filters)

    return filtered_results.head(top_n)[['Name', 'Genres_edited', 'Type', 'Origin_year', 'anime_avg_rating', 'Popularity_adjusted', 'image_url', 'synopsis']]

def get_discover_animes(top_n=20):
    if anime.empty: return pd.DataFrame()
    divided_df = get_divided_opinion_animes(anime, divided_opinion_ids)
    divided_df = divided_df.merge(anime_agg[['anime_id', 'anime_avg_rating']], left_on='MAL_ID', right_on='anime_id', how='left')
    if top_n > len(divided_df): top_n = len(divided_df)
    return divided_df.sample(n=top_n)[['Name', 'Genres_edited', 'Type', 'Origin_year', 'anime_avg_rating', 'Popularity_adjusted', 'image_url', 'synopsis']]
