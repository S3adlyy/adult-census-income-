<<<<<<< HEAD
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
=======
# ✦ PRISMA · ML STUDIO

**Un seul dépôt GitHub, une interface Django, six pipelines indépendants.**

Projet de groupe basé sur le dataset Adult (UCI) : six travaux de ML réunis dans une plateforme web commune. C'est un **squelette exécutable**, sans modèles entraînés ni dataset sensible/volumineux dans le dépôt. L'interface indique « À connecter » tant que le pipeline n'existe pas ; elle n'affiche aucune fausse prédiction.

## Les 6 modules

| Dossier ML | Objectif | Type |
|---|---|---|
| `income` | Catégorie de revenu | Classification |
| `occupation` | Segmentation des profils (occupation pour interprétation) | Clustering |
| `age` | Âge | Régression |
| `hours` | Heures de travail / semaine | Régression |
| `education` | Niveau supérieur / non supérieur | Classification binaire |
| `workclass` | Catégorie d'emploi | Classification |

> Les formulaires des 5 autres modules sont **provisoires** tant que leurs pipelines et features n'ont pas été communiqués. Seul `education` reprend le contrat de 7 champs bruts du modèle précédemment entraîné. Le dataset Adult est historique et américain ; l'outil est pédagogique et ses prédictions ne déterminent aucun accès réel à l'éducation ou à l'emploi.

## 1 · Installer et démarrer

Prérequis : Python 3.13 conseillé pour compatibilité avec le modèle education du livrable précédent, Git. Django 5.2 LTS est compatible.

**macOS / Linux**

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python manage.py migrate
python manage.py check
python manage.py test
python manage.py runserver
```

**Windows PowerShell**

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python manage.py migrate
python manage.py check
python manage.py test
python manage.py runserver
```

Ouvrir **http://127.0.0.1:8000/**. La page d'accueil et les six formulaires fonctionnent sans modèle installé. En l'absence de `.joblib`, une tentative de prédiction affiche un message d'intégration (pas un résultat fictif).

## 2 · Structure du dépôt

```text
PRISMA_ML_Team_Starter/
├── manage.py
├── config/                     # settings, routes Django
├── portal/
│   ├── catalog.py              # noms des modules et schémas de formulaire
│   ├── forms.py                # formulaire dynamique commun
│   ├── views.py                # accueil, prédiction, API, health
│   ├── urls.py
│   └── tests.py
├── ml_modules/
│   ├── common.py               # charge les pipelines .joblib (trusted only)
│   ├── income/adapter.py
│   ├── occupation/adapter.py
│   ├── age/adapter.py
│   ├── hours/adapter.py
│   ├── education/adapter.py
│   └── workclass/adapter.py
├── ml_models/                  # modèles locaux, ignorés par Git
│   └── <slug>/pipeline.joblib  # ajouter manuellement
├── templates/portal/           # UI commune, Django templates
├── static/portal/              # CSS + JS sans dépendances frontend
├── .github/workflows/          # CI (tests sans modèle)
├── CONTRIBUTING.md
├── requirements.txt
└── requirements-ml.example.txt
```

## 3 · Connecter son modèle, étape par étape

**Chaque membre suit le même protocole :**

1. Exporter son **pipeline complet entraîné** avec prétraitement, imputation et encodage (`joblib.dump(pipeline, 'pipeline.joblib')`).
2. Copier ce pipeline de source fiable dans `ml_models/<slug>/pipeline.joblib` (le fichier n'est PAS commité).
3. Installer les mêmes versions Python + ML que pendant l'entraînement. `requirements-ml.example.txt` est une **base indicative**, pas un lockfile garanti compatible.
4. Adapter **EN PRIORITÉ** `portal/catalog.py` (l'entrée de son module) et `ml_modules/<slug>/adapter.py` : champs, mapping, modalités, affichage du résultat.
5. Tester depuis l'interface `/modules/<slug>/` et depuis l'API.
6. Ajouter des tests unitaires du modèle *dans une suite séparée* si artefact disponible. La CI sans modèle reste verte.
7. Faire Pull Request.

### Exemple réel : modèle `education`

`ml_modules/education/adapter.py` attend :

```json
{
  "age": 28,
  "hours_per_week": 40,
  "capital_gain": 0,
  "capital_loss": 0,
  "workclass": "Private",
  "occupation": "Adm-clerical",
  "income": "<=50K"
}
```

Il convertit `hours_per_week` → `hours-per-week`, `capital_gain` → `capital-gain`, `capital_loss` → `capital-loss` pour le pipeline du projet précédent, puis utilise un seuil 0,52. Le pipeline du projet précédent utilise `classes_ = [0,1]` avec 1 pour « Niveau supérieur ». Installer aussi scikit-learn/XGBoost/joblib/pandas selon les versions du modèle. **Attention :** seul le pipeline exact compatible avec ces sept champs peut fonctionner directement.

### Exemple minimal générique d'adapter

```python
from ml_modules.common import predict_generic

def predict(profile):
    return predict_generic('income', profile, kind='classification')
```

Ce code fonctionne **uniquement** si le pipeline income attend exactement les mêmes noms et types de colonnes que les champs du formulaire. Sinon, mapper/transformer dans l'adapter — pas dans `portal/views.py`.

### Un modèle de clustering

Le module `occupation` regroupe les profils par similarité ; il ne prédit PAS `occupation`. Un adapter renvoie par exemple `{'label':'Segment 2','segment':'2', 'note':'...'}` ; l'étiquette et l'interprétation sont définies par le responsable du clustering.

## 4 · API commune

| URL | Méthode | Fonction |
|---|---|---|
| `/` | GET | Dashboard de l'équipe |
| `/modules/<slug>/` | GET + POST | Formulaire et résultat |
| `/api/modules/<slug>/predict/` | POST JSON | Prédiction du module |
| `/health/` | GET | Statut de l'application |

Réponses : `200` modèle exécuté ; `400` validation JSON ; `404` module inconnu ; `503` modèle non connecté / erreur de prédiction.

**CSRF est actif** pour les POST. Pour AJAX, utiliser le cookie CSRF fourni par Django après une visite sur la page et l'en-tête `X-CSRFToken`. Ne pas publier une API anonyme sans authentification et limitation de débit.

## 5 · Workflow GitHub recommandé

```bash
git init
git add .
git commit -m "chore: initialize shared Prisma ML Django skeleton"
git branch -M main
git remote add origin https://github.com/ORGANISATION/REPO.git
git push -u origin main
```

Remplacer `ORGANISATION/REPO` par le dépôt réel, préalablement créé (de préférence privé tant que le projet est en développement).

Chaque membre :

```bash
git clone https://github.com/ORGANISATION/REPO.git
cd REPO
git switch -c feature/education    # nom propre au membre
# modifier son adapter et sa configuration
git add ml_modules/education/adapter.py portal/catalog.py
git commit -m "feat(education): connect inference adapter"
git push -u origin feature/education
# ouvrir une Pull Request sur GitHub
```

Voir [CONTRIBUTING.md](CONTRIBUTING.md) et le modèle de Pull Request.

## 6 · Vérifications

```bash
python -m compileall -q config portal ml_modules
python manage.py check
python manage.py test
```

Le pipeline ML réel n'est **pas** inclus ; les tests Django utilisent des mocks pour ne pas dépendre des gros modèles.

## 7 · Production (après intégration des modèles)

Configurer `DJANGO_SECRET_KEY`, `DJANGO_DEBUG=0`, `DJANGO_ALLOWED_HOSTS`, HTTPS, PostgreSQL ou une base adaptée, WhiteNoise ou serveur de fichiers statiques, Gunicorn/Uvicorn, sécurité CSRF et cookies, journalisation sans données privées, limitation de débit, tests sur des profils réels, suivi des erreurs et de la dérive des modèles. Exécuter `python manage.py check --deploy` avec les variables de production. Ce squelette est conçu pour **développement collaboratif**, pas un déploiement public sans durcissement.

## 8 · Respect des personnes

Ne pas commiter les datasets, les modèles lourds, des identifiants ou des données personnelles. L'interface ne stocke pas les profils en base par défaut. Ne pas automatiser de décisions d'emploi, d'études ou de crédit à partir d'un modèle expérimental.
>>>>>>> 67f0886 (v1.0.0)
