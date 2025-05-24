# 📌 Introduction

The goal of this project is to develop and evaluate recommendation systems for Amazon baby products, predicting user ratings based on previous interactions and product metadata. Two datasets were used:

- `ratings_Baby.csv`: user, item, rating, timestamp
- `meta_Baby.json`: ASIN, title, description, categories

We implemented and compared three approaches:

1. **Item-based Collaborative Filtering**
2. **Content-based Filtering** (using product metadata)
3. **Matrix Factorization (SVD)** with hyperparameter tuning

These models were tested under high sparsity and metadata inconsistency, reflecting real-world data challenges.
# 📊 Data Exploration

We begin with descriptive statistics and visualizations to understand the dataset:

- **No missing values** were detected.
- The dataset is **imbalanced** with a strong bias toward 5-star ratings.
- Sparsity is extremely high, with only **0.0027%** of the user-item matrix filled.

Visualizations include:

- Rating distribution
- Ratings per user/item
- Timestamp trends
- Boxplot and average rating histograms
# 📈 Preprocessing & Splitting

We applied the following steps before modeling:

- Removed users with only 1 rating
- Filtered out items with missing metadata
- Random split: **80% train**, **10% eval**, **10% test**

The test and eval sets were stripped of ratings to simulate the real-world prediction scenario.
# 🔁 Model 1: Item-Based Collaborative Filtering

This model computes item-to-item similarity using a sparse item-user matrix and cosine similarity.

Due to the extreme sparsity (0.002733%), the similarity matrix was precomputed to avoid runtime errors.

The prediction function computes a weighted average of user ratings for similar items.

**Test Results:**
- RMSE: `1.8023`
- MAE: `0.8563`

While reasonably accurate, the model suffers from cold-start and low coverage for unseen items.
# 🧾 Model 2: Content-Based Filtering (TF-IDF)

This model uses TF-IDF vectorization on item metadata (titles + truncated descriptions).

Challenges:
- Descriptions too long → truncated to 50 words
- Full dataset caused crashes → sampled 60,000 items
- Metadata often noisy or duplicated

Cosine similarity was used to find top-k similar items rated by the user. If no similar items were found, the user mean rating was used as fallback.

**Test Results (top_k=5):**
- RMSE: `1.8999`
- MAE: `1.0137`
- Valid predictions: ~77,000/82,000
# 🔢 Model 3: Matrix Factorization (SVD)

Using the `Surprise` library, we implemented an SVD model with:

- Latent factors to model hidden user/item relationships
- Grid search over `n_factors`, `lr_all`, `reg_all`, and `n_epochs`
- 3-fold cross-validation

This model works well with sparse data and does not depend on metadata.

**Test Results:**
- RMSE: `1.1937`
- MAE: `0.9317`

This was the most accurate model overall, offering strong generalization and robustness.
# ✅ Model Comparison

| Model                         | RMSE   | MAE    |
|------------------------------|--------|--------|
| Item-Based Collaborative     | 1.8023 | 0.8563 |
| Content-Based (TF-IDF)       | 1.8999 | 1.0137 |
| Matrix Factorization (SVD)   | **1.1937** | **0.9317** |

SVD clearly outperformed other approaches in terms of accuracy.

Content-based was the weakest due to noisy metadata, despite full prediction coverage.

Collaborative filtering worked reasonably but struggled with unseen items.
# 📌 Conclusion & Future Work

We explored various recommendation approaches and iteratively refined preprocessing and modeling to manage:

- High **sparsity** and cold-start problems
- **Metadata inconsistencies** in product descriptions
- **Performance limitations** with large datasets

The SVD model offered the best trade-off between accuracy and generalization.

### 🔮 Future Directions:

- Hybrid recommendation (e.g., CF + content)
- Better metadata embeddings (e.g., BERT or SBERT)
- UI integration (e.g., Streamlit) for end-user testing
