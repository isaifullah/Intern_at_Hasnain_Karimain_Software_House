# ============================================================
# FASTAPI PRODUCT RECOMMENDATION API
# ============================================================

from fastapi import FastAPI

# FIXED IMPORT (IMPORTANT for package structure)
from app.recommender import (
    get_all_products,
    search_products,
    recommend_products,
    recommend_by_preferences
)

# ============================================================
# CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="Product Recommendation API",
    description="AI-Powered Product Recommendation System using TF-IDF and Cosine Similarity",
    version="1.0"
)

# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Product Recommendation API is Running"
    }

# ============================================================
# GET ALL PRODUCTS
# ============================================================

@app.get("/products")
def products():
    return {
        "products": get_all_products()
    }

# ============================================================
# SEARCH PRODUCTS
# ============================================================

@app.get("/search")
def search(query: str):
    return {
        "results": search_products(query)
    }

# ============================================================
# RECOMMEND PRODUCTS (CONTENT-BASED)
# ============================================================

@app.get("/recommend/{product_name}")
def recommend(product_name: str):
    return {
        "recommendations": recommend_products(product_name)
    }

# ============================================================
# USER PREFERENCE RECOMMENDATION
# ============================================================

@app.get("/recommend/user/{preferences}")
def user_recommendation(preferences: str):
    return {
        "recommendations": recommend_by_preferences(preferences)
    }