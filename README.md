# Projet Wine - Classification de vins

Ce projet utilise le jeu de données *Wine* de Scikit-learn afin d'analyser des caractéristiques chimiques de vins et de prédire leur classe.

## Contenu du projet

- **wineA.py** : Analyse des données, corrélations et graphiques

- **wineB.py** : Prédiction de classe et gestion des valeurs manquantes

- **wineC.py** : Réduction de dimension (PCA, LDA) et comparaison des performances

- **wine_bonusD.py** : Test de robustesse des modèles avec validation croisée

## Objectif

Comprendre comment des techniques d'apprentissage automatique peuvent :

- détecter des relations entre variables,

- prédire la classe d'un vin,

- gérer les valeurs manquantes,

- améliorer la précision avec des méthodes de réduction de dimension.

## Principaux résultats

- Certaines variables de ce dataset sont fortement corrélées.

- Le modèle KNN permet de prédire efficacement la classe d'un vin.

- LDA combiné avec KNN donne la meilleure performance de prédiction.

## Exécution du projet

Lancer les scripts avec Python :

python wineA.py

python wineB.py

python wineC.py

python reports/wine_bonusD.py

## Technologies

Python, NumPy, Pandas, Matplotlib, Scikit-learn

