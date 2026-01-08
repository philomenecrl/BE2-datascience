## BE2 DataScience - Système de Recherche de Documents Scientifiques
Ce projet construit un moteur de recherche hybride combinant analyse lexicale, sémantique et structurelle pour le classement de documents scientifiques.

### Structure du projet

- `perfect_matches.ipynb` : Workflow complet, de l'exploration des données à l'optimisation du modèle hybride.
- `browsers.py` : Moteurs de recherche (lexical et sémantique), fonctions d'évaluation (Précision, Rappel, F1, AUROC) et prédiction.
- `graph.py` : Construction du graphe de citations et calcul du score d'autorité PPR (Personalized PageRank).
- `utils.py` : Chargement des données, prétraitement textuel, gestion des embeddings et fonctions de visualisation.

### Méthodologie
Le système suit un pipeline en trois étapes :

1. Récupération sémantique : Identification des candidats via embeddings (all-MiniLM-L6-v2).
2. Analyse structurelle : Calcul de l'importance des documents dans le réseau de citations (PPR).
3. Score Hybride : Fusion pondérée des scores sémantiques et structurels pour un classement optimal.