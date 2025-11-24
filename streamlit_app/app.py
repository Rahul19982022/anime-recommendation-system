
import streamlit as st
import pandas as pd
from recommender import (
    get_recommendations_by_name, 
    get_discover_animes, 
    anime, 
    genres_list,
    get_anime_id_from_name,
    get_anime_details,
    anime_agg
)

# --- 1. UI Configuration ---
st.set_page_config(layout="wide", page_title="Anime Recommendation System")

# --- 2. CUSTOM CSS ---
st.markdown("""
<style>
/* --- FIX #1: GLOBAL LINK HIDER --- */
[data-testid="stHeaderActionElements"] {
    display: none !important;
}
h1, h2, h3 {
    pointer-events: none !important;
    text-decoration: none !important;
}
h1 a, h2 a, h3 a {
    display: none !important;
}

/* --- FIX #2: Rose Pink Buttons (Primary Actions Only) --- */
[data-testid="stSidebarUserContent"] button[kind="primary"] {
    background-color: #C21E56 !important; /* Rose Pink */
    color: white !important;              /* White Text */
    border: none !important;
    font-weight: 900 !important;          /* Max Font Weight */
    font-size: 16px !important;           /* Larger Text */
    text-shadow: 0.3px 0px 0px black;     
    transition: 0.2s;
    
    /* Allow wrapping for primary buttons */
    white-space: normal !important;       
    height: auto !important;              
    padding-top: 0.5rem !important;
    padding-bottom: 0.5rem !important;
    line-height: 1.2 !important;
}
[data-testid="stSidebarUserContent"] button[kind="primary"]:hover {
    background-color: #A01848 !important; /* Darker Rose on hover */
    color: white !important;
    border: none !important;
}
[data-testid="stSidebarUserContent"] button[kind="primary"] svg {
    fill: white !important;
    color: white !important;
    stroke: white !important; 
    stroke-width: 1px;
}

/* --- FIX #3: Secondary Button Styling --- */
/* Ensure standard buttons (like popovers) handle text properly */
[data-testid="stSidebarUserContent"] button[kind="secondary"] {
    white-space: normal !important;
    height: auto !important;
    padding-top: 0.2rem !important;
    padding-bottom: 0.2rem !important;
    line-height: 1.2 !important;
}

/* Main app styling */
[data-testid="stAppViewContainer"] > section:first-of-type {
    padding-top: 1rem !important;
}
[data-testid="stSidebarUserContent"] {
    padding-top: 1.5rem !important;
}

/* Recommendation Box Styling */
[data-testid="stBorderedContainer"] {
    border: 1px solid #2c2f38 !important; 
    border-radius: 10px !important;       
    padding: 1rem !important;
    margin-bottom: 1rem !important;
}
/* Title inside the box */
.rec-title {
    margin-top: 0rem !important;
    padding-top: 0rem !important;
    margin-bottom: 0.5rem;
    color: #fafafa;
    font-size: 1.25rem;
    font-weight: 600;
}
/* Poster box */
.poster-box {
    width: 150px;
    height: 210px;
    background-size: cover;
    background-position: center center;
    border-radius: 8px;
    border: 1px solid #444;
    margin-bottom: 1.25rem;
}
/* Popover spacing */
[data-testid="stPopoverBody"] h3 {
    margin-bottom: 0.25rem;
}
[data-testid="stPopoverBody"] p {
    margin-bottom: 0.75rem;
}
</style>
""", unsafe_allow_html=True)


# --- 3. Title ---
st.title("🎬 Anime Recommendation System")
st.write("Created by Rahul Goyal. A deployable recommendation engine.")

# --- 4. Sidebar ---
# --- SECTION 1: SEARCH ---
st.sidebar.header("Find Recommendations")

# --- THIS IS THE ONLY CHANGE ---
# Create the list of searchable names
all_names = list(anime['Name'].unique())
english_names = list(anime[anime['English name'] != 'Unknown']['English name'].unique())
search_options = sorted(list(set(all_names + english_names)))

# Use selectbox instead of text_input
anime_name = st.sidebar.selectbox(
    "Enter or Select an Anime:",
    options=search_options,
    index=None,
    placeholder="Type to search..."
)
# --- END OF CHANGE ---

rec_type = st.sidebar.selectbox(
    "Choose a Recommendation Type:",
    (
        "Model-Based Similarity",
        "Content (Plot) Similarity",
        "Combined Model + Genre"
    )
)

# Popover Button (Standard style with arrow)
with st.sidebar.popover("ⓘ What does this do?"):
    if rec_type == "Model-Based Similarity":
        st.markdown("**Model-Based Similarity:**")
        st.write("Finds anime that other users rated in a similar way. This is a good 'if you liked this, you might also like...' feature based on the tastes of thousands of users.")
    
    elif rec_type == "Content (Plot) Similarity":
        st.markdown("**Content (Plot) Similarity:**")
        st.write("Finds animes with similar plotlines or similar themes.")
    
    elif rec_type == "Combined Model + Genre":
        st.markdown("**Combined Model + Genre:**")
        st.write("First finds anime with similar rating patterns, then filters that list to only show ones that also have good genre similarity with the input anime genres.")


show_input_details = st.sidebar.checkbox("Show details for input anime", value=True)

genre_threshold = 0.5 
if rec_type == "Combined Model + Genre":
    genre_threshold = st.sidebar.slider(
        "Min. Genre Similarity:", 0.0, 1.0, 0.5, 0.05,
        help="This is the minimum cosine similarity required with the input anime's genres."
    )
top_n_search = st.sidebar.slider("Number of recommendations:", 5, 20, 10, key="search_slider")

st.sidebar.markdown("---")
st.sidebar.subheader("Filter Your Results (Optional)")
type_options = ['TV', 'Movie', 'OVA', 'Special', 'ONA']
all_genres = sorted(genres_list)
min_year = int(anime['Origin_year'].min())
max_year = int(anime['Origin_year'].max())

genres_preferred = st.sidebar.multiselect("Must include all of these genres:", all_genres)
type_preferred = st.sidebar.multiselect("Must be one of these types:", type_options, default=type_options)
min_anime_rating = st.sidebar.slider("Minimum average user rating:", 0.0, 10.0, 0.0, 0.1)
origin_year_range = st.sidebar.slider("Origin Year:", min_year, max_year, (min_year, max_year))

search_button = st.sidebar.button("Get Recommendations", type="primary")


# --- SECTION 2: DISCOVER ---
st.sidebar.markdown("---")
st.sidebar.header("Discover")

# --- FIX: Stacked Layout ---
# Main action button
discover_button = st.sidebar.button("Show 'Divided Opinion' Anime", type="primary")

# Small info button on the next line
with st.sidebar.popover("ℹ️ Info"): 
    st.markdown("**Divided Opinion Anime:**")
    st.write("""
    These are polarizing anime. A similar number of users loved them (rating 8+) as hated them (rating < 5). 
    
    We'll show you a random selection from this pool for you to discover!
    """)

top_n_discover = st.sidebar.slider("Number of Animes:", 5, 30, 10, key="discover_slider")


# --- 5. Store Filters ---
user_filters = {
    "Genres_preferred": genres_preferred if genres_preferred else None,
    "Type_preferred": type_preferred if type_preferred else None,
    "min_anime_rating": min_anime_rating,
    "Origin_year_range": origin_year_range
}


# --- 6. Main Page Display Logic ---
def display_recommendations(recommendations_df, is_input_anime=False):
    """Helper function to display results in a nice layout."""
    
    if is_input_anime:
        row = recommendations_df.iloc[0] 
        st.markdown(f'<h3 class="rec-title">Details for Input Anime: {row["Name"]}</h3>', unsafe_allow_html=True)
        
        with st.container(border=True):
            col1, col2 = st.columns([1, 4]) 
            with col1:
                if row['image_url'] and row['image_url'] != "NOT_FOUND":
                    image_style = f"background-image: url('{row['image_url']}')"
                else:
                    image_style = "background-image: url('https://via.placeholder.com/150x210.png?text=No+Poster')"
                st.markdown(f'<div class="poster-box" style="{image_style}"></div>', unsafe_allow_html=True)
            with col2:
                st.write(f"**Type:** {row['Type']}  |  **Year:** {row['Origin_year']}  |  **Avg. Rating:** {row['anime_avg_rating']:.2f}")
                st.write(f"**Genres:** {row['Genres_edited'].replace('|', ', ')}")
                if pd.notna(row['synopsis']):
                    with st.expander("Show Synopsis", expanded=False): 
                        st.write(row['synopsis'])
    
    else:
        for i, row in recommendations_df.reset_index(drop=True).iterrows():
            with st.container(border=True):
                st.markdown(f'<h3 class="rec-title">{i + 1}. {row["Name"]}</h3>', unsafe_allow_html=True)
                col1, col2 = st.columns([1, 4]) 
                with col1:
                    if row['image_url'] and row['image_url'] != "NOT_FOUND":
                        image_style = f"background-image: url('{row['image_url']}')"
                    else:
                        image_style = "background-image: url('https://via.placeholder.com/150x210.png?text=No+Poster')"
                    st.markdown(f'<div class="poster-box" style="{image_style}"></div>', unsafe_allow_html=True)
                with col2:
                    st.write(f"**Type:** {row['Type']}  |  **Year:** {row['Origin_year']}  |  **Avg. Rating:** {row['anime_avg_rating']:.2f}")
                    st.write(f"**Genres:** {row['Genres_edited'].replace('|', ', ')}")
                    if pd.notna(row['synopsis']):
                        with st.expander("Show Synopsis"):
                            st.write(row['synopsis'])

# --- Updated logic for which button was pressed ---
if search_button:
    if anime_name:
        anime_details = get_anime_details(anime_name, anime, anime_agg) 
        
        if anime_details is None:
            st.error("Anime is not found, please check the name and try again")
        else:
            if show_input_details:
                display_recommendations(anime_details, is_input_anime=True)
                st.markdown("---") 

            with st.spinner('Searching for the best recommendations...'):
                recommendations = get_recommendations_by_name(
                    anime_name, 
                    rec_type, 
                    top_n_search, 
                    genre_threshold=genre_threshold, 
                    **user_filters
                )
                
                if recommendations is not None and not recommendations.empty:
                    st.success(f"Here are the top {len(recommendations)} recommendations for '{anime_name}':")
                    display_recommendations(recommendations, is_input_anime=False)
                else:
                    st.error(f"No recommendations found for '{anime_name}' with the selected filters. Try broadening your search!")
    else:
        st.warning("Please enter an anime name.")

elif discover_button:
    with st.spinner("Finding controversial anime..."):
        divided_animes = get_discover_animes(top_n=top_n_discover)
        if not divided_animes.empty:
            st.success(f"🤔 Here are {len(divided_animes)} 'Divided Opinion' animes for you:")
            display_recommendations(divided_animes, is_input_anime=False)
        else:
            st.error("No 'Divided Opinion' animes found.")
            
else:
    st.info("Choose an option from the sidebar to get started!")
