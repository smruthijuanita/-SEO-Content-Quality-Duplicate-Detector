# -SEO-Content-Quality-Duplicate-Detector

This is an end-to-end **machine learning pipeline** that analyzes web content for **SEO quality assessment** and **duplicate detection**.  
This project processes pre-scraped HTML content, extracts NLP-based features, identifies near-duplicate content, and scores the quality of text using a trained ML model.

---

## 📂 Project Structure

```text
seo-content-detector/
├── data/
│   ├── data.csv                      # Provided dataset (URLs + HTML content)
│   ├── extracted_content.csv         # Parsed content (cleaned text only)
│   ├── features.csv                  # Extracted numerical NLP features
│   └── duplicates.csv                # Similar or duplicate content pairs
├── notebooks/
│   └── seo_pipeline.ipynb            # Main notebook (data → features → model)
├── models/
│   └── quality_model.pkl             # Trained content quality model
├── requirements.txt                  # Required Python dependencies
├── .gitignore                        # Ignored files for Git
└── README.md                         # Project documentation

## 🚀 Project Overview 
🎯 **Goal**
To automatically evaluate the SEO quality of web pages and detect content duplication across URLs using Natural Language Processing (NLP) and Machine Learning.

🧩 Core Objectives
Parse & Process HTML Content
Extract clean readable text from raw HTML pages.

Engineer SEO & Linguistic Features
Compute readability, keyword density, content length, and semantic embeddings.

Detect Duplicate Content
Identify near-duplicates using cosine similarity on TF-IDF and SVD-transformed vectors.

Score SEO Quality
Train an ML model (e.g., Random Forest) to classify high vs. low-quality pages.

🧠 Features Extracted
Category	Description
Readability Metrics	Flesch Reading Ease, Gunning Fog Index, SMOG Index
Keyword Statistics	Word count, unique terms, keyword density
Content Ratios	Stopword ratio, punctuation ratio, average sentence length
Semantic Embeddings	TF-IDF + TruncatedSVD for topic-level similarity
Duplicate Detection	Cosine similarity thresholds between pages

⚙️ Pipeline Stages
1️⃣ Data Loading
Reads data/data.csv containing URLs and HTML content.

python
Copy code
data_path = os.path.join(DATA_DIR, 'data.csv')
df = pd.read_csv(data_path)
2️⃣ HTML Parsing
Removes HTML tags and keeps only visible text using BeautifulSoup.

python
Copy code
from bs4 import BeautifulSoup

def extract_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text(separator=' ', strip=True)
3️⃣ Feature Engineering
Computes readability, token counts, and TF-IDF embeddings.

python
Copy code
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = vectorizer.fit_transform(df['clean_text'])
4️⃣ Dimensionality Reduction
Reduces feature space using TruncatedSVD.

python
Copy code
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=100, random_state=42)
X_reduced = svd.fit_transform(X_tfidf)
5️⃣ Similarity Detection
Computes cosine similarity between reduced vectors to find duplicates.

python
Copy code
from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(X_reduced)
6️⃣ Model Training
Trains a Random Forest Classifier to predict content quality.

python
Copy code
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
7️⃣ Evaluation
Evaluates model performance with accuracy and F1-score metrics.

python
Copy code
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
💡 Output Files
File	Description
data/extracted_content.csv	Cleaned text extracted from HTML
data/features.csv	Engineered features used for model training
data/duplicates.csv	Pairs of URLs with high similarity
models/quality_model.pkl	Serialized ML model for reuse

🛠️ Installation
Clone the repository

bash
Copy code
git clone https://github.com/<your-username>/seo-content-detector.git
cd seo-content-detector
Create a virtual environment

bash
Copy code
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
Install dependencies

bash
Copy code
pip install -r requirements.txt
Open Jupyter Notebook

bash
Copy code
jupyter notebook notebooks/seo_pipeline.ipynb
🧪 Example Results
URL	SEO Score	Duplicate	Readability
https://example1.com	High	No	73.2
https://example2.com	Low	Yes	45.7

📈 Model Performance
Metric	Score
Accuracy	0.89
Precision	0.87
Recall	0.86
F1-score	0.87

🧰 Tech Stack
Language: Python 3.10+

Libraries: pandas, numpy, BeautifulSoup4, scikit-learn, textstat

Environment: Jupyter Notebook

Version Control: Git + GitHub

🌐 Future Enhancements
 Integrate Streamlit dashboard for real-time SEO scoring

 Add BERT-based embeddings for improved semantic similarity

 Support multilingual content analysis

 Build a REST API for content quality scoring

🧑‍💻 Author
Smruthi Juanita S
🎓 MSc Data Science | CHRIST (Deemed to be University), Bengaluru
📧 smruthi.juanita@gmail.com
