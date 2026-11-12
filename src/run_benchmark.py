import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


# ==============================================================================
# EVALUATION METRICS SUITE (Predictive + Top-K Ranking + Beyond-Accuracy)
# ==============================================================================
def evaluate_recommender_metrics(test_df, preds, k=5, relevance_threshold=8.0):
    """
    Computes 8 comprehensive evaluation metrics for recommender systems:
      1. RMSE (Root Mean Squared Error)
      2. MAE (Mean Absolute Error)
      3. HitRate@1Star (% predictions within +/- 1.0 of true rating)
      4. Precision@K (Fraction of top-K recommended items that are relevant)
      5. Recall@K (Fraction of relevant items captured in top-K)
      6. F1-Score@K (Harmonic mean of Precision@K and Recall@K)
      7. NDCG@K (Normalized Discounted Cumulative Gain for ranking quality)
      8. Catalog Coverage (% of unique test catalog surfaced in top-K lists)
    """
    df_eval = test_df.copy()
    df_eval['pred'] = preds
    y_true = df_eval['Book-Rating'].values
    y_pred = df_eval['pred'].values

    # 1. Rating Prediction Error Metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    within_1_star = np.mean(np.abs(y_true - y_pred) <= 1.0) * 100.0

    # 2. Top-K Ranking & Retrieval Metrics per User
    precisions, recalls, ndcgs = [], [], []
    recommended_items = set()

    for u, group in df_eval.groupby('User-ID'):
        if len(group) < 3:
            continue
        ranked = group.sort_values('pred', ascending=False)
        top_k = ranked.head(k)
        recommended_items.update(top_k['ISBN'].tolist())

        n_rel_total = np.sum(group['Book-Rating'].values >= relevance_threshold)
        if n_rel_total == 0:
            continue

        n_rel_top_k = np.sum(top_k['Book-Rating'].values >= relevance_threshold)
        precisions.append(n_rel_top_k / min(k, len(top_k)))
        recalls.append(n_rel_top_k / n_rel_total)

        # NDCG@K calculation
        gains = (2.0 ** top_k['Book-Rating'].values - 1.0)
        discounts = np.log2(np.arange(2, len(top_k) + 2))
        dcg = np.sum(gains / discounts)

        ideal_ranked = group.sort_values('Book-Rating', ascending=False).head(k)
        ideal_gains = (2.0 ** ideal_ranked['Book-Rating'].values - 1.0)
        idcg = np.sum(ideal_gains / discounts)
        if idcg > 0:
            ndcgs.append(dcg / idcg)

    p_at_k = np.mean(precisions) * 100.0
    r_at_k = np.mean(recalls) * 100.0
    f1_at_k = (2 * p_at_k * r_at_k / (p_at_k + r_at_k)) if (p_at_k + r_at_k) > 0 else 0.0
    ndcg_at_k = np.mean(ndcgs) * 100.0
    coverage = (len(recommended_items) / df_eval['ISBN'].nunique()) * 100.0

    return {
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'HitRate_1Star_%': round(within_1_star, 2),
        'Precision@5_%': round(p_at_k, 2),
        'Recall@5_%': round(r_at_k, 2),
        'F1@5_%': round(f1_at_k, 2),
        'NDCG@5_%': round(ndcg_at_k, 2),
        'Catalog_Coverage_%': round(coverage, 2)
    }


def main():
    print("Loading Book-Crossing dataset...")
    books = pd.read_csv('Books.csv', low_memory=False)
    ratings = pd.read_csv('Ratings.csv')

    explicit = ratings[ratings['Book-Rating'] > 0].copy()
    u_counts = explicit['User-ID'].value_counts()
    b_counts = explicit['ISBN'].value_counts()

    dense = explicit[
        explicit['User-ID'].isin(u_counts[u_counts >= 30].index) &
        explicit['ISBN'].isin(b_counts[b_counts >= 30].index)
    ]
    valid_books = books[books['ISBN'].isin(dense['ISBN'].unique())].drop_duplicates('ISBN').set_index('ISBN')
    dense = dense[dense['ISBN'].isin(valid_books.index)].copy()

    print(f"Filtered Dense Subset: {len(dense):,} ratings | {dense['User-ID'].nunique():,} users | {dense['ISBN'].nunique():,} books")
    train_df, test_df = train_test_split(dense, test_size=0.20, random_state=42)

    mu = train_df['Book-Rating'].mean()
    u_means = train_df.groupby('User-ID')['Book-Rating'].mean()
    i_means = train_df.groupby('ISBN')['Book-Rating'].mean()
    i_counts = train_df.groupby('ISBN')['Book-Rating'].count()

    # 1. Bayesian Popularity
    m_q = i_counts.quantile(0.50)
    bayesian_scores = (i_counts / (i_counts + m_q)) * i_means + (m_q / (i_counts + m_q)) * mu
    pop_preds = [np.clip(bayesian_scores.get(r['ISBN'], mu), 1, 10) for _, r in test_df.iterrows()]

    # 2. Content-Based Filtering (TF-IDF)
    valid_books['content'] = (
        valid_books['Book-Title'].fillna('') + ' ' +
        (valid_books['Book-Author'].fillna('') + ' ') * 2 +
        valid_books['Publisher'].fillna('')
    )
    tfidf_mat = TfidfVectorizer(stop_words='english', max_features=3500, ngram_range=(1, 2)).fit_transform(valid_books['content'])
    cb_sim_df = pd.DataFrame(cosine_similarity(tfidf_mat), index=valid_books.index, columns=valid_books.index)

    pivot_u_i = train_df.pivot(index='User-ID', columns='ISBN', values='Book-Rating')
    u_row_means = pivot_u_i.mean(axis=1)
    pivot_centered = pivot_u_i.sub(u_row_means, axis=0).fillna(0)

    cb_preds = []
    for _, row in test_df.iterrows():
        u, i = row['User-ID'], row['ISBN']
        base_u = u_means.get(u, mu)
        if u in pivot_u_i.index and i in cb_sim_df.index:
            u_rated = pivot_u_i.loc[u].dropna()
            sims = cb_sim_df.loc[i, u_rated.index]
            top_k = sims[sims > 0.05].nlargest(15)
            if len(top_k) > 0:
                devs = u_rated.loc[top_k.index] - base_u
                pred = base_u + np.dot(top_k.values, devs.values) / (np.sum(top_k.values) + 1e-8)
            else:
                pred = base_u + (i_means.get(i, mu) - mu) * 0.5
        else:
            pred = base_u
        cb_preds.append(np.clip(pred, 1, 10))

    # 3. User-to-User CF
    u2u_sim_df = pd.DataFrame(cosine_similarity(pivot_centered), index=pivot_u_i.index, columns=pivot_u_i.index)
    u2u_preds = []
    for _, row in test_df.iterrows():
        u, i = row['User-ID'], row['ISBN']
        base_u = u_means.get(u, mu)
        if u in u2u_sim_df.index and i in pivot_u_i.columns:
            i_raters = pivot_u_i[i].dropna()
            sims = u2u_sim_df.loc[u, i_raters.index]
            top_k = sims[sims > 0].nlargest(20)
            if len(top_k) > 0:
                devs = i_raters.loc[top_k.index] - u_row_means.loc[top_k.index]
                pred = base_u + np.dot(top_k.values, devs.values) / (np.sum(top_k.values) + 1e-8)
            else:
                pred = base_u + (bayesian_scores.get(i, mu) - mu)
        else:
            pred = base_u
        u2u_preds.append(np.clip(pred, 1, 10))

    # 4. Item-to-Item CF
    i2i_sim_df = pd.DataFrame(cosine_similarity(pivot_centered.T), index=pivot_u_i.columns, columns=pivot_u_i.columns)
    i2i_preds = []
    for _, row in test_df.iterrows():
        u, i = row['User-ID'], row['ISBN']
        base_u = u_means.get(u, mu)
        if u in pivot_u_i.index and i in i2i_sim_df.index:
            u_rated = pivot_u_i.loc[u].dropna()
            sims = i2i_sim_df.loc[i, u_rated.index]
            top_k = sims[sims > 0].nlargest(20)
            if len(top_k) > 0:
                devs = u_rated.loc[top_k.index] - base_u
                pred = base_u + np.dot(top_k.values, devs.values) / (np.sum(top_k.values) + 1e-8)
            else:
                pred = base_u + (bayesian_scores.get(i, mu) - mu)
        else:
            pred = base_u
        i2i_preds.append(np.clip(pred, 1, 10))

    # 5. Matrix Factorization (Biased FunkSVD)
    user_ids, isbn_ids = sorted(train_df['User-ID'].unique()), sorted(train_df['ISBN'].unique())
    u2idx = {u: idx for idx, u in enumerate(user_ids)}
    i2idx = {b: idx for idx, b in enumerate(isbn_ids)}

    np.random.seed(42)
    n_factors, lr, reg, epochs = 25, 0.006, 0.06, 30
    b_u, b_i = np.zeros(len(user_ids)), np.zeros(len(isbn_ids))
    P = np.random.normal(0, 0.08, (len(user_ids), n_factors))
    Q = np.random.normal(0, 0.08, (len(isbn_ids), n_factors))

    samples = [(u2idx[r['User-ID']], i2idx[r['ISBN']], r['Book-Rating']) for _, r in train_df.iterrows()]
    for _ in range(epochs):
        np.random.shuffle(samples)
        for u_idx, i_idx, r in samples:
            err = r - (mu + b_u[u_idx] + b_i[i_idx] + np.dot(P[u_idx], Q[i_idx]))
            b_u[u_idx] += lr * (err - reg * b_u[u_idx])
            b_i[i_idx] += lr * (err - reg * b_i[i_idx])
            P_old = P[u_idx].copy()
            P[u_idx] += lr * (err * Q[i_idx] - reg * P[u_idx])
            Q[i_idx] += lr * (err * P_old - reg * Q[i_idx])

    svd_preds = []
    for _, row in test_df.iterrows():
        u, i = row['User-ID'], row['ISBN']
        if u in u2idx and i in i2idx:
            pred = mu + b_u[u2idx[u]] + b_i[i2idx[i]] + np.dot(P[u2idx[u]], Q[i2idx[i]])
        elif u in u2idx:
            pred = mu + b_u[u2idx[u]]
        elif i in i2idx:
            pred = mu + b_i[i2idx[i]]
        else:
            pred = mu
        svd_preds.append(np.clip(pred, 1, 10))

    # 6. Hybrid Ensemble
    hybrid_preds = np.clip(0.60 * np.array(svd_preds) + 0.25 * np.array(i2i_preds) + 0.15 * np.array(cb_preds), 1, 10)

    models = {
        'Hybrid Ensemble (SVD + I2I + CB)': hybrid_preds,
        'Matrix Factorization (Biased SVD)': svd_preds,
        'Content-Based (TF-IDF)': cb_preds,
        'Bayesian Popularity': pop_preds,
        'User-to-User CF (KNN)': u2u_preds,
        'Item-to-Item CF (Adjusted Cosine)': i2i_preds,
    }

    eval_rows = []
    for name, preds in models.items():
        metrics = evaluate_recommender_metrics(test_df, preds, k=5, relevance_threshold=8.0)
        metrics['Paradigm'] = name
        eval_rows.append(metrics)

    cols = ['Paradigm', 'RMSE', 'MAE', 'HitRate_1Star_%', 'Precision@5_%', 'Recall@5_%', 'F1@5_%', 'NDCG@5_%', 'Catalog_Coverage_%']
    eval_df = pd.DataFrame(eval_rows)[cols].sort_values('RMSE')
    print("\n=== COMPREHENSIVE EVALUATION METRICS BENCHMARK ===")
    print(eval_df.to_string(index=False))

if __name__ == '__main__':
    main()
