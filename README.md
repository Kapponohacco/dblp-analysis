# DBLP Dataset Analysis, Recommendation & RAG Prototype

A comprehensive analysis pipeline for the [DBLP](https://dblp.org/) computer science bibliography dataset. This project demonstrates structured data analysis, NLP-driven topic discovery, content-based recommendation, and retrieval-augmented generation (RAG).

---

## 📋 Table of Contents

---

## Overview


### Subset Strategy


---

## Setup

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- ~4GB free disk space (for dataset + processed files)
- Optional: CUDA-capable GPU (for faster embedding generation)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Rekrutacja_nokia
   ```

2. **Install dependencies:**
   This project uses `uv` for fast dependency management.
   ```bash
   uv sync
   ```
   This will create a `.venv` virtual environment and install all required packages.

3. **Set up environment variables:**
   Create a `.env` file in the root directory for the RAG features (Notebook 05):
   ```env
   GOOGLE_API_KEY=your_google_api_key
   # OPENAI_API_KEY=your_openai_api_key
   ```

4. **Run the Notebooks:**
   Start the Jupyter environment and execute the notebooks sequentially, starting with `01_data_ingestion.ipynb` to download and parse the dataset:
   ```bash
   uv run jupyter notebook
   ```

## Project Structure

```
Rekrutacja_nokia/
├── README.md                           
├── pyproject.toml                      # Project config & dependencies
├── requirements.txt                    # Pip-compatible requirements (alternative)
├── .env                                # Environment variables - GOOGLE_API_KEY,OPENAI_API_KEY
├── .gitignore
│
├── data/                               # Data directory (not in git)
│   ├── dblp.xml.gz                     # Raw compressed dataset (~1GB)
│   ├── dblp.dtd                        # XML DTD schema
│   └── processed/
│       ├── dblp_subset.parquet         # Processed subset
│       ├── embeddings_sample.npy       # Cached embeddings (clustering)
│       └── embeddings_recommender.npy  # Cached embeddings (recommender)
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb         # Download, parse, preprocess
│   ├── 02_eda_basic.ipynb              # Papers/year, venues, authors, keywords
│   ├── 03_topic_clustering.ipynb       # Embeddings + HDBSCAN + UMAP
│   ├── 04_recommendation.ipynb         # Content-based recommender
│   └── 05_rag_layer.ipynb              # RAG prototype
│
└── src/
    ├── __init__.py
    ├── parser.py                       # Streaming XML parser
    ├── preprocessing.py                # Text cleaning & tokenization
    ├── network.py                      # Co-author graph analysis
    ├── embeddings.py                   # Sentence embedding generation
    ├── clustering.py                   # HDBSCAN + UMAP clustering
    ├── recommender.py                  # Content-based recommendation
    └── rag.py                          # RAG pipeline (optional)
```

---

## Methodology

### Data Ingestion (Notebook 01)
- Extracts: title, authors, year, venue, publication typey
- Saves to Apache Parquet for fast subsequent loading

### Exploratory Data Analysis (Notebooks 02)
- **Papers per year**: Time series + growth rate analysis
- **Venue trends**: Conference vs journal proportions over time
- **Author productivity**: Top-N ranking + Lotka's Law verification (power-law distribution)
- **Co-authorship network**: NetworkX graph with degree/betweenness centrality
- **Keyword analysis**: TF-IDF on titles, unigram + bigram frequency, word clouds

### Topic Discovery (Notebook 03)
- **Embeddings**: `all-MiniLM-L6-v2` sentence-transformer (384-dim)
- **Dimensionality reduction**: UMAP (superior local structure preservation vs PCA)
- **Clustering**: HDBSCAN — no need to pre-specify k, handles noise and varying density
- **Summary**: Manual analysis of clusters based on titles and keywords, using LLM to help summarize clusters and infer topics.

### Recommendation System (Notebook 04)
- **Content-based filtering** using cosine similarity over title embeddings
- Supports: title search, free-text query, author lookup, hybrid (title + co-authorship)
- Quality evaluation via similarity score distribution analysis

### RAG Layer (Notebook 05)
- **Retrieve**: Embedding-based paper retrieval
- **Augment**: Structured context formatting for LLM prompts
- **Generate**: Multi-backend support (OpenAI, Google Gemini)
- All the LLM responses combined at the end for a final report.

---

## Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| XML Parsing | `lxml.iterparse` | Memory-efficient streaming |
| Data Storage | Pandas + Parquet | Fast columnar access |
| Embeddings | `sentence-transformers` | State-of-the-art semantic similarity |
| Clustering | HDBSCAN + UMAP | No preset k, handles noise |
| Network Analysis | NetworkX | Standard graph library |
| Visualization | Matplotlib + Plotly | Static & interactive |
| Package Management | uv | Fast, modern Python tooling |

---

## License

This project uses the DBLP dataset, which is available under the [DBLP terms of use](https://dblp.org/xml/).
