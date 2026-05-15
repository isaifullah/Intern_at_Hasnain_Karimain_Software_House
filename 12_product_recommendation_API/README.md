
---
title: Product Recommendation API
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# 🎯 Product Recommendation API (FastAPI + Machine Learning)

An AI-powered Product Recommendation System built using FastAPI, TF-IDF Vectorization, and Cosine Similarity.  
It recommends similar products based on product search or user preferences.

This project simulates real-world recommendation engines used in e-commerce platforms like Amazon.

---

# 🚀 Features

- 🔍 Product Search API
- 🎯 Content-Based Recommendation System
- 🧠 TF-IDF + Cosine Similarity Model
- 👤 User Preference-Based Recommendations
- 📦 Product Listing API
- ⚡ FastAPI High-Performance Backend
- 📚 Auto-generated Swagger Documentation (/docs)

---

# 🧠 Tech Stack

- Python 🐍
- FastAPI ⚡
- Scikit-learn 🤖
- Pandas 📊
- NumPy 🔢
- Uvicorn 🚀

---

# 📁 Project Structure

product_recommendation_api/<br>
├── app/<br>
│   ├── main.py              # FastAPI backend<br>
│   ├── recommender.py       # Recommendation logic<br>
│   ├── data/<br>
│   │   └── products.csv     # Dataset<br>
│<br>
├── requirements.txt<br>
├── Dockerfile<br>
├── README.md<br>

---

# ⚙️ How It Works

1. Data Processing  
   We combine Product Name, Category, and Sub-category into a single text feature.

2. Feature Extraction  
   TF-IDF Vectorization converts text into numerical vectors.

3. Similarity Calculation  
   Cosine Similarity measures similarity between products.

4. Recommendation  
   Top similar products are returned via API endpoints.

---

# 📡 API Endpoints

## 🏠 Root
GET /

Response:
{
  "message": "Product Recommendation API is running"
}

---

## 📦 Get All Products
GET /products

---

## 🔍 Search Products
GET /search?query=watch

---

## 🎯 Recommend Products
GET /recommend/{product_name}

Example:
GET /recommend/iPhone 14

---

## 👤 User Preference Recommendation
GET /recommend/user/{preferences}

Example:
GET /recommend/user/smartwatch bluetooth

---

# 🧪 Example Response

{
  "recommendations": [
    {
      "name": "Noise Smartwatch",
      "category": "Accessories",
      "score": 0.91
    },
    {
      "name": "Fire-Boltt Smartwatch",
      "category": "Accessories",
      "score": 0.88
    }
  ]
}

---

# 🚀 Deployment

Option 1: Hugging Face Spaces (Recommended)  
https://huggingface.co/spaces

Steps:
- Create a Space
- Upload project files
- Choose Docker
- Auto deploy

---

# 🧠 Key Concepts

- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Cosine Similarity
- REST API Development
- Machine Learning Deployment

---

# 🔥 Future Improvements

- Collaborative Filtering
- Deep Learning Embeddings (BERT)
- User Authentication (JWT)
- Database Integration (MongoDB/PostgreSQL)
- Frontend UI (React / Streamlit)
- Caching with Redis

---

# 👨‍💻 Author

Saif Ullah  
AI/ML Engineer | FastAPI Developer | Recommendation Systems Enthusiast

---

# ⭐ Support

If you like this project, please give it a ⭐ on GitHub.
