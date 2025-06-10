# 💡 Two-Tower Deep Learning Model for Personalized Banking Recommendations

🚀 **Deep Learning-Based Two-Tower Architecture for Personalized Financial Services**

(Inspired by Meta’s Deep Learning Recommendation Model – DLRM)

---

## 📌 Project Overview  
This project implements a deep learning-based **Two-Tower architecture**, inspired by large-scale industry models such as Meta’s DLRM (Deep Learning Recommendation Model), to deliver personalized banking product recommendations.

Each tower encodes different data modalities **customer features** and **product attributes** into a shared embedding space. This allows the system to efficiently match customers with suitable financial products based on learned latent representations.

By leveraging modern deep learning techniques, the system captures **complex behavioral and financial patterns**, enhancing recommendation accuracy and boosting customer engagement.

---

## 📂 Dataset  
- **Customer Features:** Demographics, financial metrics, behavioral scores, credit info  
- **Product Features:** Product type, risk level, yield, duration, terms  
- **Total Samples:** 10,000+ customer-product data points  

---

## 🏗️ Model Architecture

The model adopts a **Two-Tower Deep Neural Network**, inspired by large-scale recommender systems like Meta’s DLRM and Google’s YouTube Retrieval system.

### 🏛️ Customer Tower
- **Inputs:** Age, income, credit utilization, loan amount, digital engagement, etc.  
- **Architecture:** Dense layers + Batch Normalization + Dropout + ReLU activations  
- **Output:** Fixed-length **customer embedding vector** representing latent user preferences  

### 🏦 Product Tower
- **Inputs:** Product category, risk level, expected yield, duration, financial requirements  
- **Architecture:** Symmetrical to customer tower for parallel learning  
- **Output:** **Product embedding vector** capturing financial product characteristics  

### 🔗 Matching Layer
- Computes **cosine similarity** between customer and product embeddings  
- High similarity → strong customer-product match  
- Can be extended with cross features or interaction layers for complex relationships  

---

## 🧠 Methodology

### 📊 1. Feature Engineering
- **Customer Features:** Standardized continuous features (e.g., income), encoded categorical features (e.g., gender, engagement tier)  
- **Product Features:** Embedded categorical fields like risk category, duration type  
- **Normalization** ensures stable and efficient learning  

### ⚙️ 2. Model Training
- Implemented using **TensorFlow Functional API** for flexibility  
- **Loss Function:** Contrastive loss or Triplet loss to maximize relevant pair similarity  
- **Negative Sampling** used to improve discriminative learning  

### 🎯 3. Recommendation Logic
- For a given customer, similarity scores are computed for all available products  
- **Top-K ranking** used to retrieve the most relevant recommendations  
- Optional filters: risk-tolerant vs conservative, interest-specific, etc.  

### 📈 4. Evaluation
Evaluation metrics include:
- ✅ Mean Reciprocal Rank (MRR)  
- ✅ Precision@K / Recall@K  
- ✅ Normalized Discounted Cumulative Gain (nDCG)  

---

## 🌐 Streamlit Interface

To make the model interactive and usable by business teams, the project includes a **Streamlit dashboard**.

### 🔧 Key Features:
- Upload or choose a customer profile  
- Visualize customer attributes and financial behavior  
- Get **Top-5 personalized banking product recommendations**  
- Interactive insights:
  - Similarity scores
  - Matching product details
  - Customer-product embedding proximity
- **SHAP visualizations** for transparency and explainability  

### ▶️ To Run the App:
```bash
streamlit run Recommendation_Dashboard.py
