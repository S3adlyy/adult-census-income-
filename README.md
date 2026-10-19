# Rapport ML — classification binaire du niveau d'études

## Objectif métier
> Aider les personnes à trouver le bon parcours de formation en estimant leur niveau d'études à partir de leur profil socio-professionnel, afin de leur proposer dès le départ une orientation adaptée.

Cible opérationnelle : `Niveau non supérieur` (Bac compris) vs `Niveau supérieur` (toute expérience post-bac, y compris non diplômante). Ce groupe ne signifie **pas** que la personne possède un diplôme supérieur.

## Dataset : audit des données

- Source : `adult-all.csv`, Adult Census américain historique. 48,842 lignes × 15 colonnes, 52 doublons identiques supprimés => 48,790 observations.
- `?` traités comme des manquants : workclass 2,799, occupation 2,809, native-country 857. Pas d'imputation effectuée avant les splits.
- 2 classes : non supérieur 22,169 (45.44 %), supérieur 26,621 (54.56 %). **SMOTE non justifié** dans ce cas ; `class_weight` testé et non retenu.
- Outliers IQR : âge 215, heures/semaine 13486, capital-gain 4035, capital-loss 2282. Des IQR nuls sur les capitaux et des distributions professionnelles asymétriques rendent une suppression automatique incorrecte.
- Association avec `education` : corrélations numériques point-bisériales peu fortes (`age` -0.0201, heures 0.0692, gain de capital 0.0729). Pour les catégories, Cramér V (occupation 0.4528, income 0.2354, workclass 0.1454). Corrélation et association n'impliquent pas causalité.
- **Répétitions de profils** : 22,731 combinaisons de caractéristiques distinctes parmi 48,789 lignes ; 3,081 combinaisons sont liées à des niveaux d'études contradictoires. Les informations disponibles ne peuvent donc pas déterminer parfaitement la cible.

## Anti-fuite de données et protocole

- Variables prédites exclues des features : `education`, `education-num` ; `fnlwgt` supprimée ; `race`, `sex`, ainsi que d'autres données personnelles, non utilisés. Features finales : `age`, `hours-per-week`, `capital-gain`, `capital-loss`, `workclass`, `occupation`, `income`.
- **Pipeline** : imputation médiane / catégorie `Unknown`, `log1p` des capitaux, standardisation pour les modèles linéaires/SVM, OneHotEncoder(handle_unknown='ignore'), classifieur. Tous les ajustements sont refaits **uniquement sur le pli train**.
- Séparation stratifiée, graine 42 : 34 152 train (70 %), 7 319 validation (15 %), 7 319 test final (15 %). Sélection sur CV à 5 plis sur train. Test verrouillé jusqu'à la fin.

## Comparaison réelle des modèles : F1 macro CV, 5 plis

| Algorithme | F1 macro CV | Écart-type | ROC-AUC CV |
|---|---:|---:|---:|
| XGBoost | 0.7234 | 0.0062 | 0.7974 |
| HistGradientBoosting | 0.7233 | 0.0059 | 0.7967 |
| LightGBM | 0.7231 | 0.0056 | 0.7972 |
| RandomForest | 0.7219 | 0.0056 | 0.7942 |
| RandomForest balanced | 0.7217 | 0.0056 | 0.7941 |
| DecisionTree | 0.7161 | 0.0063 | 0.7858 |
| LogisticRegression | 0.7078 | 0.0070 | 0.7739 |
| LinearSVM | 0.7059 | 0.0074 | 0.7738 |
| LogisticRegression balanced | 0.7024 | 0.0054 | 0.7739 |
| Dummy majoritaire | 0.3530 | 0.0000 | 0.5000 |

**Remarque** : les différences entre XGBoost, LightGBM et HistGradientBoosting sont très faibles (moins de 0,001 en F1 macro CV) : il ne faut **pas** prétendre à une supériorité statistique importante de XGBoost.

## Optimisation du gagnant (XGBoost)

- RandomizedSearchCV : 12 combinaisons tirées aléatoirement, 5 plis, optimisation de `f1_macro` sur train. Meilleur F1 CV : 0.7241.
- Meilleurs hyperparamètres : `{"model__subsample": 1.0, "model__reg_lambda": 3, "model__n_estimators": 200, "model__min_child_weight": 10, "model__max_depth": 4, "model__learning_rate": 0.05}`.
- Validation tenue à part pour choisir le seuil : `0.52` (F1 macro visé). Pas de tuning sur test.

## Résultats mesurés sur le test final (7 319 observations)

| Mesure | Valeur |
|---|---:|
| Accuracy (avec seuil validé) | 0.7323 |
| Balanced accuracy (avec seuil validé) | 0.7316 |
| Precision supérieur (avec seuil validé) | 0.7627 |
| Recall supérieur (avec seuil validé) | 0.7395 |
| F1 macro (avec seuil validé) | **0.7308** |
| ROC-AUC | **0.8049** |
| Average precision (PR-AUC) | 0.8419 |
| Brier score (proba, plus faible est mieux) | 0.1789 |
| F1 macro entraînement (seuil par défaut) | 0.7282 |
| F1 macro test (seuil par défaut) | 0.7307 |

**Matrice de confusion à seuil 0.52**, lignes = vérité, colonnes = prédiction, ordre `[Niveau non supérieur, Niveau supérieur]` :

| | Prédit non sup. | Prédit sup. |
|---|---:|---:|
| Réel non sup. | 2407 | 919 |
| Réel sup. | 1040 | 2953 |

La courbe d'apprentissage est dans `figures/courbe_apprentissage.png`. Écart train / test du F1 macro au seuil par défaut : -0.0026 (ne montre pas d'overfitting marqué). Attention : le test reste **insuffisant** pour orienter de façon automatisée sans confirmation de l'utilisateur.

## Expériences de robustesse et disponibilité des caractéristiques

| Variables | F1 macro CV (modèle XGBoost optimisé, train seulement) |
|---|---:|
| Toutes les variables | 0.7241 |
| Sans income | 0.7223 |
| Profil minimal sans emploi | 0.6033 |

Le revenu annuel ajoute peu ici, mais la profession apporte une très grande part du signal. Sans profession/revenu, les performances baissent à ~0,60. Cela confirme un problème de **transfert à des étudiants sans expérience professionnelle**. Les profils identiques réservés dans un stress-test par groupe atteignent F1 macro 0.7334 (test distinct, apprentissage indépendant ; ne pas confondre avec le test principal).

## Limites scientifiques et métier

1. **Dataset et population** : Adult est historique et américain, non représentatif des étudiants tunisiens, des personnes sans emploi ou des diplômés actuels.
2. **Objectif** : le label supérieur signifie "avoir fréquenté le post-bac", pas l'obtention certifiée d'un diplôme. Un parcours de formation personnalisé nécessite compétences, préférences, langue, acquis et conditions d'admission.
3. **Équité** : ne pas inclure race/sex ne suffit pas à garantir l'absence de biais indirects via occupation, revenu ou statut professionnel. Un audit par sous-groupe serait nécessaire.
4. **Fiabilité** : accuracy ≈ 73 % et erreurs assez nombreuses. Ne pas refuser ou imposer une formation uniquement sur la prédiction.
5. **Calibration** : un Brier score est calculé, mais la calibration n'a pas fait l'objet d'un ajustement spécifique. Afficher "estimation probabiliste" plutôt que "certitude".
6. **Représentativité de l'entrée** : un formulaire Django doit demander exactement les 7 features attendues ; si celles-ci ne sont pas collectables, il faut reconstituer un dataset centré étudiants et réentraîner.

## Fichiers et déploiement technique

- `Projet_ML_Adult_Orientation_Optimise.ipynb` : notebook indépendant, cellules documentées, de bout en bout, reproduit le nettoyage, la comparaison, l'optimisation et la sauvegarde.
- `train_orientation.py` : même entraînement en ligne de commande.
- `modele_orientation_pipeline.joblib` : **pipeline entraîné** (imputation/one-hot/log1p + XGBoost).
- `model_metadata.json` : caractéristiques, classes, seuil 0,52.
- `predict_orientation.py` : `predire_profil(profil)` prédit à partir des champs CSV bruts.
- `api_fastapi.py` : API facultative `/health`, `/predict`.
- `django_integration/` : exemple de vue / route Django, ajuster les imports et les protections (auth, CSRF, droits, logs) à votre app.
- `requirements.txt` / `requirements_entrainement.txt` : environnement figé. Chargement joblib avec une version différente de scikit-learn ou XGBoost n'est pas garanti.

### Commandes

```bash
# Environnement recommandé : Python 3.13, versions du requirements.txt
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_entrainement.txt
python train_orientation.py --data adult-all.csv --out .
python predict_orientation.py
uvicorn api_fastapi:app --host 127.0.0.1 --port 8000
```

Test API :

```bash
curl -X POST http://127.0.0.1:8000/predict \
 -H 'Content-Type: application/json' \
 -d '{"age":28,"hours-per-week":40,"capital-gain":0,"capital-loss":0,"workclass":"Private","occupation":"Adm-clerical","income":"<=50K"}'
```

Dans Django : ajouter le répertoire du modèle au chemin Python, importer `predire_profil` dans la vue, configurer la route, garder `joblib` et `model_metadata.json` en lecture seule. Ne jamais charger de fichiers joblib non fiables. Faire une vérification de sécurité et de protection des données pour l'application réelle.

**Statut** : *techniquement intégrable et évalué sur Adult ; validation métier externe encore nécessaire avant un déploiement réel d'orientation automatique*.
