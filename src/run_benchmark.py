import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

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

    y_true = test_df['Book-Rating'].values
    results = []
    for name, preds in models.items():
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        mae = mean_absolute_error(y_true, preds)
        acc_1 = np.mean(np.abs(y_true - np.array(preds)) <= 1.0) * 100
        results.append({'Paradigm': name, 'RMSE': round(rmse, 4), 'MAE': round(mae, 4), 'Within_1_Star_%': round(acc_1, 2)})

    print(pd.DataFrame(results).to_string(index=False))

if __name__ == '__main__':
    main()
