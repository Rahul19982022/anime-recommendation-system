# 🎌 Anime Recommendation System

This project implements **10+ anime recommendation strategies** spanning collaborative filtering, content-based methods, genre analysis, and behavioral insights.  
For public demonstration, a Streamlit app showcases a subset of these techniques.  
For visualization or brief explanation of all the strategies implemented, please refer to the [📊 Presentations](#-presentations) section below.

---

## 🔍 Project Overview

This repository contains **two complementary components**:

1. **Full Recommendation System (Research & Experiments)**  (`recommendation_features.ipynb file`)  
   Implemented in Google Colab, this version explores **10+ recommendation strategies**, consisting both of user-dependent and user-independent strategies to recommend anime.

2. **Streamlit Demo App (Public Deployment)**  
   A lightweight, publicly accessible demo showcasing **4 user-independent recommendation features**, deployed on Streamlit Cloud.

This separation ensures both **depth (research)** and **accessibility (public demo)**.

---

## 🚀 Streamlit Demo App

🔗 **Live App:** https://anime-recommender-rahul.streamlit.app/

The Streamlit app demonstrates recommendation techniques that **do not depend on private user data**, making it suitable for public sharing.

### ✨ Features in the Demo
- Anime–Anime similarity using embeddings
- Content / plot similarity using plot embeddings  
- Hybrid: Model + Genre Similarity Recommendations
- Divisive/Controversial Anime Recommendations
  
### 🛠️ App Architecture

```
streamlit_app/
│── app.py
│── recommender.py
│── requirements.txt

app_data/
│── *.pkl / *.csv
```

---

## 🏗️ Deployable App Builder

The notebook **`Deployable_App_Builder.ipynb`** bridges the research system and the deployable app.

It is used to:
- Generate lightweight, preprocessed datasets required for the Streamlit app to function correctly, saved in the app_data folder
- Build `app.py` and `recommender.py` for the Streamlit app’s user interface and backend recommendation logic

---

## 📂 Repository Structure

```
anime-recommendation-system/
│
├── recommendation_features.ipynb
├── Deployable_App_Builder.ipynb
│
├── archives/
│   ├── 1_data_preprocessing.ipynb
│   ├── 2_collaborative_filtering_model.ipynb
│   └── 3_create_plot_embeddings.ipynb
│
├── datasets/
│   ├── created_datasets/
│   └── kaggle_dataset/
│
├── model/
├── app_data/
├── streamlit_app/
├── requirements.txt
└── README.md
```

---

## 🚀 Features Implemented (Full System)

Below are the key recommendation strategies implemented in this project:

1. **Rating Prediction-based Recommendations** (Collaborative Filtering using Deep Learning)
2. **User-User Similarity via Embeddings**
3. **Anime-Anime Similarity via Embeddings**
4. **Genre Similarity-based Recommendations**
5. **Hybrid: Model + Genre Similarity Recommendations**
6. **Content Matching using Plot Embeddings**
7. **Personalized Genre Preference Scoring**
8. **User-User Similarity via Genre Preferences**
9. **Watch History-based User Similarity**
10. **Origin-Year-based Gap Recommendations**
11. **Divisive/Controversial Anime Recommendations**

📌 *For visualization or brief explanation of each feature, refer to the [📊 Presentations](#-presentations) section below.*


## ⚙️ Setup Instructions (Google Colab)

1. First, open a Colab notebook and mount the drive using following command in a cell.

```python
from google.colab import drive
drive.mount('/content/drive')
```
   Now we will clone the repository using following command in a cell.  
   You need to **change the save path according to your Google Drive** i.e. where you will save the project.  
 
```python
import os
#replace the save_path variable with your drive path 
save_path = '/content/drive/MyDrive/'
os.chdir(save_path)
!git clone https://github.com/Rahul19982022/anime-recommendation-system.git
```

This will create `anime_recommendation_system` folder. Close the notebook and open that folder.

2. Follow the instructions in the next section i.e. 'Required Data & Models' section to download the required datasets and model files.  
   Place them in their respective folders as described.

3. Run `recommendation_features.ipynb`.  
   The `os.chdir(proj_path)` command is used inside the notebook to navigate to the project folder.  
   You only need to **change the folder path according to your Google Drive** location of the project.

---

## 📦 Required Data & Models

Download the required files from the links below and place them in the corresponding folders:

- 📁 [Preprocessed Datasets (`created_datasets`)](https://drive.google.com/drive/folders/1_bVHyoS_7fgeE5EvjHh4aUqIxiIdcJrs?usp=sharing) → `datasets/created_datasets/`  
  - ✅ Required for running the **main notebook**

- 📁 [Trained Model File](https://drive.google.com/file/d/1dbSTKyevwdZk-SEpiOmMD8EbfDoz-RrV/view?usp=sharing) → `model/`  
  - ✅ Required for running the **main notebook**

- 📁 [Raw Dataset from Kaggle](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020) → `datasets/kaggle_dataset/`  
  - 🟡 *Optional – only needed if you want to run the archived notebooks for preprocessing or training*

---

## ⚠️ Environment Note

This project was developed and tested in **Google Colab (Python 3.11)**.

A `requirements.txt` is provided as reference for the packages used, and mentioned versions if required 

---

## 📓 How to Use

1. Run `recommendation_features.ipynb` in Google Colab, it is recommended to run the entire notebook first, which will take about 9-10 minutes.
2. Then you can experiment with a recommendation method by calling its function 
3. I have included one example under every recommendation method so you can refer to it
4. Get anime recommendations 🎉

---

## 📊 Presentations

- 📑 [Main Presentation](https://docs.google.com/presentation/d/1qagcPzebKpr_LADMoHzZHwbdMbKxJqcA2sNKGam1iDU/edit?usp=sharing)  
  *(Visual overview of all features)*

- 📘 [Explanation Slides](https://docs.google.com/presentation/d/1_etxR5wh607hY8e4ZFN_ceOlVaMKY3XR0BkdPfJTEGA/edit?usp=drive_link)  
  *(Brief textual summary for every feature – for reference)*

---

## 🙌 Acknowledgements
- Content embeddings are generated using the model [`Alibaba-NLP/gte-large-en-v1.5`](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5) from Hugging Face.
- Dataset: [Kaggle Anime Recommendation Database](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020)  
- Libraries: `tensorflow`, `pandas`, `numpy`, `scikit-learn`, `sentence-transformers`