import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

wine = load_wine()
X = wine.data
y = wine.target

pipe_poly1_logreg = Pipeline([
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=1, include_bias=False)),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

pipe_knn_fixed = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

scores_poly = cross_val_score(pipe_poly1_logreg, X, y, cv=cv, scoring="accuracy")
scores_knn_fixed = cross_val_score(pipe_knn_fixed, X, y, cv=cv, scoring="accuracy")

print("=== BONUS D : Robustesse (CV 5-fold) ===")
print(f"Poly(degree=1)+LogReg -> acc moyenne={scores_poly.mean():.3f} ± {scores_poly.std():.3f}")
print(f"KNN(k=5)              -> acc moyenne={scores_knn_fixed.mean():.3f} ± {scores_knn_fixed.std():.3f}")

best_score = 0.0
best_params = None

for k in [1, 3, 5, 7, 9, 11]:
    for w in ["uniform", "distance"]:
        pipe_knn = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k, weights=w))
        ])
        scores = cross_val_score(pipe_knn, X, y, cv=cv, scoring="accuracy")
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_params = (k, w)

print("\nRecherche de k (KNN) pour rester en juste-apprentissage :")
print(f"Meilleurs paramètres: k={best_params[0]}, weights='{best_params[1]}'")
print(f"Accuracy CV (meilleur k): {best_score:.3f}")
