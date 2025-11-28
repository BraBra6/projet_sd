# wine_bonusD.py  — Bonus D : Robustesse (CV 5-fold)
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Jeu de données
wine = load_wine()
X = wine.data
y = wine.target

# Modèle polynomial d'ordre 1 = modèle linéaire (exigence énoncé)
pipe_poly1_logreg = Pipeline([
    ("scaler", StandardScaler()),
    ("poly1", PolynomialFeatures(degree=1, include_bias=False)),
    # Ne PAS mettre multi_class (déprécié) ; solver lbfgs par défaut → multinomial automatiquement
    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
])

# Baseline KNN pour comparaison
pipe_knn = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

# Évaluation robustesse (CV 5-fold)
scores_poly1 = cross_val_score(pipe_poly1_logreg, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
scores_knn   = cross_val_score(pipe_knn, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

print("=== BONUS D : Robustesse (CV 5-fold) ===")
print(f"Poly(degree=1)+LogReg -> acc moyenne={scores_poly1.mean():.3f} ± {scores_poly1.std():.3f}")
print(f"KNN(k=5)              -> acc moyenne={scores_knn.mean():.3f} ± {scores_knn.std():.3f}")

# Recherche d'un "juste-apprentissage" pour KNN en ajustant k
grid = GridSearchCV(
    pipe_knn,
    {"knn__n_neighbors": [1,3,5,7,9,11], "knn__weights": ["uniform","distance"]},
    cv=cv, scoring="accuracy", n_jobs=-1
)
grid.fit(X, y)
print("\nRecherche de k (KNN) pour rester en juste-apprentissage :")
print("Meilleurs paramètres:", grid.best_params_)
print(f"Accuracy CV (meilleur k): {grid.best_score_:.3f}")
