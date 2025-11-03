# -SEO-Content-Quality-Duplicate-Detector

This is an end-to-end **machine learning pipeline** that analyzes web content for **SEO quality assessment** and **duplicate detection**.  
This project processes pre-scraped HTML content, extracts NLP-based features, identifies near-duplicate content, and scores the quality of text using a trained ML model.

---

## Project Overview 
**Goal**<br>
To automatically evaluate the SEO quality of web pages and detect content duplication across URLs using Natural Language Processing (NLP) and Machine Learning.

**Core Objectives**<br>
- Parse & Process HTML Content
- Extract clean readable text from raw HTML pages.

**Engineer SEO & Linguistic Features**<br>
Compute readability, keyword density, content length, and semantic embeddings.

**Detect Duplicate Content**<br>
Identify near-duplicates using cosine similarity on TF-IDF and SVD-transformed vectors.

**Score SEO Quality**<br>
Train an ML model (e.g., Random Forest) to classify high vs. low-quality pages.

## Features Extracted
| Category                | Description                                                |
| ----------------------- | ---------------------------------------------------------- |
| **Readability Metrics** | Flesch Reading Ease, Gunning Fog Index, SMOG Index         |
| **Keyword Statistics**  | Word count, unique terms, keyword density                  |
| **Content Ratios**      | Stopword ratio, punctuation ratio, average sentence length |
| **Semantic Embeddings** | TF-IDF + TruncatedSVD for topic-level similarity           |
| **Duplicate Detection** | Cosine similarity thresholds between pages                 |


## Setup Instructions

```bash
git clone https://github.com/yourusername/seo-content-detector
cd seo-content-detector
pip install -r requirements.txt
jupyter notebook notebooks/seo_pipeline.ipynb
```

## Quick Start
1. Place your dataset (data.csv) inside the data/ folder.<br>
Each row should contain: url, html_content.
2. Run all cells in notebooks/seo_pipeline.ipynb.
3. Outputs will be generated in /data and /models folders:
- extracted_content.csv: Parsed clean text
- features.csv:  Engineered NLP features
- duplicates.csv: Similar/duplicate pairs
- quality_model.pkl: Trained ML model
4. Review model performance and duplicate detection summary at the end of the notebook.

## Key Points
**(1) Libraries Chosen**<br>
Used **BeautifulSoup4** for robust HTML parsing, **textstat** for readability scoring, and scikit-learn for modeling and similarity detection.

**(2) HTML Parsing**<br>
Extracted only visible text using BeautifulSoup’s tag-based filtering to eliminate scripts, styles, and metadata noise.

**(3) Similarity Threshold**<br>
Applied **cosine similarity > 0.85** on SVD-reduced TF-IDF embeddings to balance recall and false positives.

**(4) Model Selection**<br>
Selected **Random Forest Classifier** for its interpretability and stability on tabular NLP features.

## Results Summary
| Metric                                      | Score              |
| ------------------------------------------- | ------------------ |
| **Accuracy**                                | 0.89               |
| **F1-score**                                | 0.87               |
| **Duplicates Found**                        | 14 out of 68 pages |
| **Avg. Quality Score (High-quality pages)** | 74.3 / 100         |

**Inference:** Higher readability (Flesch > 60) and unique keyword density (1–3%) correlated strongly with SEO success.

## Limitations
- Dataset size limits generalization to large domains.
- TF-IDF embeddings lack deeper semantic context (can improve with BERT).
- Quality labels rely on heuristic thresholds, not human SEO audits.

<br>
Author: Smruthi Juanita S<br>
MSc Data Science | CHRIST (Deemed to be University), Bengaluru<br>
smruthi.juanita@msds.christuniversity.in
