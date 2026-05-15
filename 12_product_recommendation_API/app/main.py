# ============================================================
# FASTAPI PRODUCT RECOMMENDATION API WITH WEB UI
# ============================================================

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os

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
    title="🛍️ Product Recommendation API",
    description="""
    ## AI-Powered Product Recommendation System
    
    ### Features:
    - **🔍 Search**: Find products by name, category, or description
    - **🎯 Recommendations**: Get similar product suggestions
    - **👤 Personalization**: Get recommendations based on preferences
    
    ### How to Use:
    1. Use the **Web Interface** at the home page
    2. Explore **API docs** at `/docs`
    3. Test endpoints directly from this Swagger UI
    """,
    version="2.0.0"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# ROOT ENDPOINT - Serve HTML UI
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home():
    """Serve the main web interface"""
    html_file = "index.html"
    if os.path.exists(html_file):
        return FileResponse(html_file)
    else:
        # Fallback if HTML file doesn't exist
        return """
        <html>
            <body style="font-family: Arial; text-align: center; padding: 50px;">
                <h1>🛍️ Product Recommendation API</h1>
                <p>API is running successfully!</p>
                <p>📖 <a href="/docs">API Documentation</a></p>
                <p>Add index.html for the visual interface</p>
            </body>
        </html>
        """

# ============================================================
# API STATUS ENDPOINT
# ============================================================

@app.get("/api")
def api_status():
    """API status check endpoint"""
    return {
        "message": "Product Recommendation API is Running",
        "version": "2.0.0",
        "status": "healthy",
        "endpoints": {
            "web_ui": "/",
            "api_docs": "/docs",
            "products": "/products",
            "search": "/search?query=iphone",
            "recommend": "/recommend/iPhone 14 Pro Max",
            "user_recommend": "/recommend/user?preferences=gaming"
        }
    }

# ============================================================
# GET ALL PRODUCTS
# ============================================================

@app.get("/products")
def products():
    """Get all available products"""
    all_products = get_all_products()
    return {
        "total": len(all_products),
        "products": all_products
    }

# ============================================================
# SEARCH PRODUCTS
# ============================================================

@app.get("/search")
def search(
    query: str = Query(..., description="Search query", min_length=1),
    limit: Optional[int] = Query(10, description="Max results", ge=1, le=50)
):
    """Search products by name, category, or description"""
    results = search_products(query)
    
    # Apply limit if provided
    if limit and len(results) > limit:
        results = results[:limit]
    
    return {
        "query": query,
        "total_results": len(results),
        "results": results
    }

# ============================================================
# RECOMMEND PRODUCTS (CONTENT-BASED)
# ============================================================

@app.get("/recommend/{product_name}")
def recommend(
    product_name: str,
    top_n: Optional[int] = Query(5, description="Number of recommendations", ge=1, le=20)
):
    """Get product recommendations based on product name"""
    recommendations = recommend_products(product_name)
    
    # Apply top_n limit
    if top_n and len(recommendations) > top_n:
        recommendations = recommendations[:top_n]
    
    return {
        "product": product_name,
        "total_recommendations": len(recommendations),
        "recommendations": recommendations
    }

# ============================================================
# USER PREFERENCE RECOMMENDATION
# ============================================================

@app.get("/recommend/user")
def user_recommendation(
    preferences: str = Query(..., description="User preferences (e.g., 'gaming console', 'wireless headphones')"),
    top_n: Optional[int] = Query(5, description="Number of recommendations", ge=1, le=20)
):
    """Get personalized recommendations based on user preferences"""
    recommendations = recommend_by_preferences(preferences)
    
    # Apply top_n limit
    if top_n and len(recommendations) > top_n:
        recommendations = recommendations[:top_n]
    
    return {
        "preferences": preferences,
        "total_recommendations": len(recommendations),
        "recommendations": recommendations
    }

# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "product-recommendation-api"
    }