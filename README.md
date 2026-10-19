
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
