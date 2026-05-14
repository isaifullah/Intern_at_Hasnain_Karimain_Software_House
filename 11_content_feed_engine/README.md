# 🎯 Personalized Content Feed Engine

A behavior-driven recommendation system that delivers personalized content suggestions based on user interactions, content similarity, and real-time preference updates.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

</div>

---

# 📖 Overview

This project implements a personalized content recommendation engine inspired by modern recommendation systems used by platforms such as Netflix, TikTok, and Google News.

The system analyzes user behavior including clicks and views, builds dynamic user preference profiles, and generates personalized recommendations using content-based filtering techniques.

The recommendation engine is built using the Microsoft MIND dataset and demonstrates practical machine learning concepts including:

- User behavior analysis
- Content-based recommendation systems
- TF-IDF vectorization
- Cosine similarity
- Time decay scoring
- Real-time profile updates
- Hybrid recommendation strategies

---

# ✨ Features

## Core Features

- Behavior-based recommendation engine
- Personalized content feeds
- Real-time interaction updates
- TF-IDF content representation
- Cosine similarity matching
- Time decay scoring for recent interactions
- User preference profiling
- Hybrid recommendation system
- Trending news recommendations
- Cold-start user handling
- Category-based filtering
- Recommendation export to CSV
- Interactive CLI interface

---

# 🏗️ System Architecture

```text
MIND Dataset
   │
   ├── News Articles
   └── User Behavior Logs
            │
            ▼
Data Processing Layer
   ├── Data Cleaning
   ├── Interaction Extraction
   └── Behavior Scoring
            │
            ▼
Feature Engineering
   ├── TF-IDF Vectorization
   ├── Similarity Matrix
   └── User Preference Profiles
            │
            ▼
Recommendation Engine
   ├── Personalized Recommendations
   ├── Trending Recommendations
   ├── Hybrid Recommendations
   └── Real-Time Updates
            │
            ▼
Output Layer
   ├── CLI Interface
   ├── Visualizations
   ├── CSV Export
   └── User Profiles
```

---

# 📊 Dataset

This project uses the **MIND (Microsoft News Dataset)** for recommendation system research.

## Dataset Statistics

| Component | Details |
|---|---|
| Articles | 50,000+ |
| User Interactions | 150,000+ |
| Categories | Multiple news categories |
| Dataset Type | Real-world news recommendation dataset |

## Dataset Structure

```text
Dataset/
├── news.tsv
├── behaviors.tsv
├── entity_embedding.vec (optional)
└── relation_embedding.vec (optional)
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/isaifullah/11_content_feed_engine.git
cd 11_content_feed_engine
```

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Add Dataset

Place the MIND dataset files inside the `Dataset/` directory.

---

# 🚀 Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebook and run all cells sequentially.

The CLI interface will launch automatically in the final section.

---

# 🧠 Recommendation Pipeline

## 1. Data Cleaning

- Removes missing values
- Removes duplicate articles
- Converts text to lowercase
- Combines title and abstract

---

## 2. Behavior Processing

User interactions are extracted from impressions and categorized as:

| Interaction | Weight |
|---|---|
| Click | 2 |
| View | 1 |

---

## 3. Time Decay Scoring

Recent interactions are weighted more heavily using exponential decay.

```text
final_score = weight × (0.5)^(days_ago / 30)
```

---

## 4. User Preference Profiling

The system builds normalized user profiles based on interaction history and content categories.

Example:

```python
{
    "sports": 0.45,
    "news": 0.30,
    "technology": 0.25
}
```

---

## 5. Content Representation

TF-IDF vectorization converts article text into numerical feature vectors.

- Maximum Features: 5000
- Stop Words Removal: Enabled
- Similarity Metric: Cosine Similarity

---

## 6. Hybrid Recommendation Strategy

The final recommendation combines personalized relevance with trending popularity.

```text
hybrid_score = 0.7 × personalized_score + 0.3 × popularity_score
```

---

# 📈 Recommendation Types

## Personalized Recommendations

Generated using:

- User interaction history
- Content similarity
- Category preferences

## Trending Recommendations

Popular content based on global engagement.

## Hybrid Recommendations

Combination of:

- Personalized content
- Trending content

## Cold-Start Recommendations

Trending content for users with little or no interaction history.

---

# 📊 Visualizations

The project includes multiple visualizations:

- User interest distributions
- Recommendation score charts
- Top news categories
- Preference analysis

Libraries used:

- Matplotlib
- Seaborn

---

# 🖥️ CLI Interface

The project includes an interactive command-line interface with the following features:

```text
1. Get personalized recommendations
2. Add new interaction
3. View user profile
4. Visualize user preferences
5. Filter recommendations by category
6. Show trending content
7. Compare recommendations
8. Export recommendations
9. Show system statistics
0. Exit
```

---

# 📁 Project Structure

```text
personalized-content-feed-engine/
│
├── Dataset/
│   ├── news.tsv
│   ├── behaviors.tsv
│   ├── entity_embedding.vec
│   └── relation_embedding.vec
│
├── recommender.ipynb
├── user_profiles.json
├── recommendations.csv
├── requirements.txt
└── README.md
```

---

# 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Storage | JSON, CSV |
| Environment | Jupyter Notebook |

---

# 🔥 Key Concepts Demonstrated

- Content-Based Filtering
- Recommendation Systems
- User Behavior Analysis
- Time Decay Modeling
- TF-IDF Feature Engineering
- Cosine Similarity
- Hybrid Recommendation Systems
- Real-Time Recommendation Updates

---

# 🚀 Future Improvements

Potential future enhancements include:

- Streamlit or Flask web interface
- Collaborative filtering
- Deep learning-based recommendations
- BERT embeddings
- Real-time streaming pipelines
- Recommendation evaluation metrics
- REST API integration

---

# 🤝 Contributing

Contributions are welcome.

Possible contribution areas:

- Performance optimization
- Additional recommendation algorithms
- UI improvements
- Evaluation metrics
- Deployment support

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

GitHub: https://github.com/isaifullah

---

# 🙏 Acknowledgments

- Microsoft Research for the MIND Dataset
- Scikit-learn
- Pandas
- NumPy
- Open-source ML community

---

<div align="center">

### Personalized Recommendation Systems Using Machine Learning

Built with Python and Scikit-learn

</div>