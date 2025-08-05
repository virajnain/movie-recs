# 🎬 Movie Recommendation System

A **collaborative filtering-based movie recommendation engine** built with **PyTorch** and deployed via **Flask**. The system predicts user ratings for movies using **embedding-based neural networks** and serves recommendations through a REST API.

---

## 🚀 Features
- **Collaborative Filtering:** Learns user and movie embeddings for personalized recommendations.  
- **Neural Network in PyTorch:** Implements user/movie embeddings with bias terms and trains using MSE loss.  
- **Data Preprocessing:** Normalizes ratings with z-score standardization for improved training stability.  
- **Flask API:** Exposes endpoints for real-time movie recommendations with CORS support.  
- **Model Persistence:** Saves and loads trained models and mappings for efficient API inference.  
- **Scalable Training:** Uses PyTorch DataLoader for efficient batching on large datasets.

---

## 🧠 How It Works
1. **Data Loading:**  
   - Reads `movies.csv` and `ratings.csv` (MovieLens format).  
   - Maps `userId` and `movieId` to integer indices.

2. **Model Training:**  
   - Defines a PyTorch `RecommenderNet` using **user and movie embeddings** + bias terms.  
   - Optimizes with **MSE loss** using Adam optimizer.  
   - Normalizes ratings and splits data into training/validation sets.

3. **API Inference:**  
   - Loads the trained model and mappings.  
   - Accepts user input and returns top-N movie recommendations via Flask API.

## 🛠 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
```bash
# Clone the repository
git clone https://github.com/virajnain/movie-recs.git
cd movie-recs

# Install dependencies
pip install flask flask-cors torch pandas scikit-learn numpy
```

##🛠 Technologies Used
- **Languages & Libraries**: Python, Pandas, NumPy, Pickle

- **Machine Learning Frameworks**: PyTorch (Neural Networks, Embeddings, DataLoader)

- **Web Frameworks**: Flask (REST API), Flask-CORS

- **Dataset**: MovieLens

- **Development Tools**: Git (Version Control)