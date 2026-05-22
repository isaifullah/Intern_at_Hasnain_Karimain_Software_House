<div align="center">

# 🔍 Image Similarity Search Engine using Deep Feature Extraction

### *An Industry-Level AI-Powered Visual Search and Semantic Retrieval System*

Developed by **Khalid Saifullah**

</div>

---

# 📖 Project Overview

The **Image Similarity Search Engine** is an advanced deep learning-based visual retrieval system designed to find visually and semantically similar images using feature extraction techniques.

This project combines:
- Deep CNN embeddings using MobileNetV2
- High-speed vector search using FAISS
- Semantic multimodal retrieval using CLIP
- Real-time image similarity matching
- Natural language text-to-image search

The system is capable of:
- Image-to-image similarity retrieval
- Semantic image understanding
- Text-based image search
- Large-scale fast vector retrieval
- Real-time deployment using Streamlit

This project demonstrates practical applications of:
- Computer Vision
- Deep Learning
- Vector Databases
- Semantic Retrieval Systems
- Multimodal AI

---

# 🚀 Key Features

## ✅ CNN-Based Feature Extraction
- Uses pretrained MobileNetV2
- Extracts deep visual embeddings
- Generates normalized feature vectors
- Optimized for image similarity tasks

---

## ✅ Image Similarity Search
Supports:
- Cosine Similarity
- Euclidean Distance
- Top-K Similar Image Retrieval

The system compares deep feature embeddings to retrieve visually similar products.

---

## ✅ FAISS Fast Vector Search

Integrated with :contentReference[oaicite:0]{index=0} for:
- High-speed nearest neighbor search
- Efficient large-scale retrieval
- Optimized vector indexing
- Faster inference compared to brute-force search

---

## ✅ CLIP Semantic Search

Integrated with :contentReference[oaicite:1]{index=1} for:
- Semantic image understanding
- Image-to-image semantic similarity
- Text-to-image retrieval
- Natural language image search

### Example Queries

```text
black sports shoes
blue denim jeans
red casual shirt
white sneakers
```

The system retrieves semantically related products using text queries.

---

## ✅ Advanced Features

- Category Filtering
- Saved Feature Databases (.pkl)
- Saved FAISS Indexes
- GPU Acceleration Support
- Streamlit Deployment Ready
- Real-Time Similarity Search
- Semantic Text Retrieval
- Industry-Level Modular Architecture

---

# 🧠 System Workflow

```text
Dataset
   ↓
Image Preprocessing
   ↓
CNN Feature Extraction
   ↓
Feature Normalization
   ↓
Feature Database Creation
   ↓
FAISS Vector Indexing
   ↓
Similarity Search
   ↓
CLIP Semantic Search
   ↓
Streamlit Deployment
```

---

# 🛠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core Development |
| TensorFlow | CNN Feature Extraction |
| PyTorch | CLIP Semantic Search |
| OpenCV | Image Processing |
| NumPy | Numerical Operations |
| Scikit-learn | Similarity Metrics |
| FAISS | Fast Vector Search |
| Transformers | CLIP Model |
| Streamlit | Web Deployment |

---

# 📂 Project Structure

```text
18_image_similarity_search_engine/
│
├── dataset/
│   ├── images/
│   │   ├── 10000.jpg
│   │   ├── 10001.jpg
│   │   ├── 10002.jpg
│   │   └── ...
│   │
│   └── styles.csv
│
├── models/
│   ├── features.pkl
│   ├── image_paths.pkl
│   ├── metadata.pkl
│   ├── faiss_index.bin
│   ├── clip_features.pkl
│   └── clip_faiss.bin
│
├── uploads/
│
├── outputs/
│   └── results/
│
├── image_similarity_search.ipynb
├── app.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## Step 1: Clone Repository

```bash
git clone https://github.com/isaifullah/image-similarity-search-engine.git
cd image-similarity-search-engine
```

---

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ⚡ GPU Installation for PyTorch

## CUDA 12.1

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## CUDA 11.8

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

# ▶️ How to Run the Project

## Run Jupyter Notebook

```bash
jupyter notebook
```

Open:
```text
image_similarity_search.ipynb
```

---

## Run Streamlit Application

```bash
streamlit run app.py
```

---

# 🔥 Similarity Search Methods

## 1️⃣ CNN Similarity Search

Uses MobileNetV2 embeddings for:
- Visual similarity matching
- Feature vector comparison
- Product retrieval

Supports:
- Cosine Similarity
- Euclidean Distance

---

## 2️⃣ FAISS Vector Search

FAISS enables:
- Fast nearest neighbor retrieval
- Large-scale vector indexing
- Real-time similarity search

---

## 3️⃣ CLIP Semantic Search

CLIP enables:
- Semantic image retrieval
- Natural language image search
- Cross-modal understanding

Example:
```text
Input Text:
"black sports shoes"

Output:
Semantically related shoe images
```

---

# ⚠️ Important Technical Notes

## TensorFlow + PyTorch Kernel Crash Issue

TensorFlow and PyTorch may crash the Jupyter kernel because both attempt to allocate GPU memory simultaneously.

### Recommended Solution

1. Run CNN feature extraction first
2. Save CNN feature vectors
3. Restart kernel
4. Run CLIP feature extraction separately

This improves memory management and prevents GPU conflicts.

---

# 🛠 Recommended Optimizations

## Create Models Directory

```python
os.makedirs("models", exist_ok=True)
```

---

## Safe Feature Normalization

```python
norm = np.linalg.norm(features)

if norm == 0:
    return None

features = features / norm
```

---

## Start with Small Dataset

```python
metadata = metadata.sample(5000, random_state=42).reset_index(drop=True)
```

Start with 5000 images before processing the complete dataset.

---

# 🌐 Streamlit Deployment

The Streamlit application supports:
- Image upload
- Similarity retrieval
- Semantic text search
- Interactive visualization
- Real-time AI inference

Deployment files include:

```text
app.py
requirements.txt
README.md
```

---

# 🎯 Final Project Capabilities

- Deep Feature Extraction
- Image Similarity Search
- Semantic Image Retrieval
- Text-to-Image Search
- FAISS Vector Search
- Large-Scale Image Indexing
- Real-Time AI Inference
- Streamlit Web Deployment

---

# 📌 Future Improvements

Possible future enhancements:
- Hybrid recommendation system
- User preference learning
- Real-time database indexing
- API integration
- Cloud deployment
- Docker containerization
- Vector database integration
- Multi-modal recommendation engine

---

# ✅ Final Status

The project architecture is scalable, modular, and industry-level.

Current implementation includes:
- CNN embeddings
- Feature normalization
- FAISS vector indexing
- CLIP semantic search
- Saved vector databases
- Category filtering
- Visualization system
- Streamlit-ready deployment

This project is suitable for:
- AI/ML portfolios
- Internship submissions
- Deep learning demonstrations
- Computer vision applications
- Semantic retrieval systems

---

<div align="center">

## ⭐ Developed by Saif ullah

### AI Engineer | Deep Learning | Computer Vision | NLP

</div>