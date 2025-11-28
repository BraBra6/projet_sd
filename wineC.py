import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def best_accuracy_baseline(X, y):
    best_score = 0
    best_k = None
    for k in [3, 5, 7]:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k))
        ])
        scores = cross_val_score(pipe, X, y, cv=cv)
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_k = k
    return best_score, best_k

def best_accuracy_pca(X, y):
    best_score = 0
    best_params = None
    n_features = X.shape[1]
    for n in [2, 3, 5, 10]:
        if n > n_features:
            continue
        for k in [3, 5, 7]:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n, random_state=42)),
                ("knn", KNeighborsClassifier(n_neighbors=k))
            ])
            scores = cross_val_score(pipe, X, y, cv=cv)
            mean_score = scores.mean()
            if mean_score > best_score:
                best_score = mean_score
                best_params = (n, k)
    return best_score, best_params

def best_accuracy_lda(X, y):
    best_score = 0
    best_params = None
    n_classes = len(np.unique(y))
    max_comp = max(1, n_classes - 1)
    for n in range(1, max_comp + 1):
        for k in [3, 5, 7]:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("lda", LDA(n_components=n)),
                ("knn", KNeighborsClassifier(n_neighbors=k))
            ])
            scores = cross_val_score(pipe, X, y, cv=cv)
            mean_score = scores.mean()
            if mean_score > best_score:
                best_score = mean_score
                best_params = (n, k)
    return best_score, best_params

def evaluate_dataset(name, X, y):
    print(f"\n=== {name} ===")
    b_score, b_k = best_accuracy_baseline(X, y)
    p_score, (p_n, p_k) = best_accuracy_pca(X, y)
    l_score, (l_n, l_k) = best_accuracy_lda(X, y)

    rows = [
        ["Baseline (Scaler+KNN)", b_score, f"k={b_k}"],
        ["PCA -> KNN", p_score, f"n_comp={p_n}, k={p_k}"],
        ["LDA -> KNN", l_score, f"n_comp={l_n}, k={l_k}"],
    ]
    df = pd.DataFrame(rows, columns=["Méthode", "Accuracy moyenne", "Paramètres"])
    df["Accuracy moyenne"] = df["Accuracy moyenne"].apply(lambda x: f"{x:.4f}")
    print(df.to_string(index=False))

wine = load_wine()
X_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
y_wine = wine.target
evaluate_dataset("Wine", X_wine, y_wine)

iris = load_iris()
X_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
y_iris = iris.target
evaluate_dataset("Iris", X_iris, y_iris)
