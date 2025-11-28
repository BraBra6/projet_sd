import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer

wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target
target_names = wine.target_names

print("Ordre des colonnes :")
for i, c in enumerate(X.columns):
    print(f"{i:02d} - {c}")

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])
pipe.fit(X, y)

def pred(sample, name):
    arr = np.array(sample).reshape(1, -1)
    y_pred = pipe.predict(arr)[0]
    proba = pipe.predict_proba(arr)[0]
    proba_dict = {target_names[i]: round(float(p), 3) for i, p in enumerate(proba)}
    print(f"\n{name}")
    print("classe prédite :", y_pred, f"({target_names[y_pred]})")
    print("probabilités   :", proba_dict)

print("\n=== A) Prédictions avec données complètes ===")
test = [11, 1, 1, 1, 100, 1, 1, 1, 1, 1, 1, 1, 111]
new_d_full = [13, 2, 2, 20, 99, 2, 2, 0.4, 2, 5, 1, 2.5, 500]
pred(test, "test")
pred(new_d_full, "new_d_full")

print("\n=== B) Imputation valeurs manquantes ===")
new_d_missing = [np.nan, np.nan, 3, 15, 80, 3, 1, 0.3, 2, 5, 1, 2.5, 500]

knn_imp = KNNImputer(n_neighbors=5)
knn_imp.fit(X)
imputed_knn = knn_imp.transform([new_d_missing])[0]
print("\nValeurs imputées (KNNImputer) :")
for c, v in zip(X.columns, imputed_knn):
    print(f"{c:25s} : {v:.3f}")
pred(imputed_knn, "new_d_missing (KNNImputer)")

med_imp = SimpleImputer(strategy="median")
med_imp.fit(X)
imputed_med = med_imp.transform([new_d_missing])[0]
print("\nValeurs imputées (médiane) :")
for c, v in zip(X.columns, imputed_med):
    print(f"{c:25s} : {v:.3f}")
pred(imputed_med, "new_d_missing (médiane)")
