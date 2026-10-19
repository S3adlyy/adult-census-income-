# livrable_orentation — Application Django + 6 modules Machine Learning

> Projet d'équipe Adult / UCI, **prêt à lancer en local avec ton vrai modèle `education`**.
> Le ZIP fourni contient l'artefact Education pour la démonstration ; les cinq autres espaces servent de base à l'intégration des autres membres.

## 1. Ce qu'il y a dans ce dossier

- `manage.py` : commandes Django (serveur, migration, tests).
- `config/` : réglages Django, URLs principales, serveur WSGI / ASGI.
- `portal/` : interface commune, formulaires, vues, routes et tests.
- `templates/portal/` : pages HTML (accueil, modules, structure commune).
- `static/portal/` : styles CSS responsive, JavaScript (navigation/recherche).
- `ml_modules/common.py` : chargeur des pipelines et helpers communs.
- `ml_modules/education/adapter.py` : **connexion directe** entre formulaire Django et modèle XGBoost.
- `ml_modules/{income,occupation,age,hours,workclass}/adapter.py` : points de branchement réservés aux autres membres.
- `ml_models/education/pipeline.joblib` : le **modèle réel entraîné** du livrable joint, qui inclut le prétraitement.
- `ml_models/education/model_metadata.json` : features attendues, classes et seuil **0.52**.
- `scripts/doctor_model.py` : vérifie le modèle et produit une prédiction, même sans Django.
- `requirements.txt` : dépendances et versions nécessaires.
- `.github/workflows/django-ci.yml` : vérification automatique des Pull Requests.

### Arborescence

```text
livrable_orentation/
├── manage.py
├── requirements.txt
├── .gitignore
├── .env.example
├── README.md
├── CONTRIBUTING.md
├── config/
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── portal/
│   ├── catalog.py
│   ├── forms.py
│   ├── views.py
│   ├── urls.py
│   ├── tests.py
│   └── context_processors.py
├── ml_modules/
│   ├── common.py
│   ├── education/adapter.py
│   ├── income/adapter.py
│   ├── occupation/adapter.py
│   ├── age/adapter.py
│   ├── hours/adapter.py
│   └── workclass/adapter.py
├── ml_models/
│   ├── education/
│   │   ├── pipeline.joblib
│   │   ├── model_metadata.json
│   │   └── exemple_profil.json
│   ├── income/
│   ├── occupation/
│   ├── age/
│   ├── hours/
│   └── workclass/
├── templates/portal/
│   ├── base.html
│   ├── home.html
│   └── module.html
├── static/portal/
│   ├── css/app.css
│   └── js/app.js
└── scripts/doctor_model.py
```

## 2. Lancer l'application sur macOS (M1 ou Intel)

Prérequis : **Python 3.13**, Terminal et accès à Internet pour installer les dépendances. Installer Python 3.13 s'il manque.

**Option rapide :** après décompression, entre `cd livrable_orentation && bash scripts/start_mac.sh` (installation, vérifications et démarrage automatiques).

**Option étape par étape dans Terminal :**

```bash
cd livrable_orentation
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Vérifie ton modèle, **avant même de démarrer Django** :

```bash
python scripts/doctor_model.py
```

Résultat de l'exemple fourni : **Niveau supérieur**, score de classe supérieure **0.5797**, seuil **0.52**.

Puis lance le site :

```bash
python manage.py check
python manage.py migrate
python manage.py test
python manage.py runserver
```

Ouvre dans le navigateur :

- Accueil : **http://127.0.0.1:8000/**
- Ton modèle Education : **http://127.0.0.1:8000/modules/education/**
- État des modules : **http://127.0.0.1:8000/health/**

Pour arrêter : **Ctrl+C** dans Terminal. Pour reprendre plus tard : `source .venv/bin/activate && python manage.py runserver`.

### Sous Windows (PowerShell)

```powershell
cd livrable_orentation
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/doctor_model.py
python manage.py migrate
python manage.py test
python manage.py runserver
```

## 3. Tester une prédiction dans le navigateur

Ouvre `/modules/education/`, remplis ces champs :

| Champ affiché | Valeur exemple | Colonne ML |
|---|---|---|
| Âge | `28` | `age` |
| Heures travaillées / semaine | `40` | `hours-per-week` |
| Gains en capital | `0` | `capital-gain` |
| Pertes en capital | `0` | `capital-loss` |
| Catégorie d'emploi | `Secteur privé` (= `Private`) | `workclass` |
| Profession | `Administration` (= `Adm-clerical`) | `occupation` |
| Revenu annuel (catégorie UCI) | `Jusqu’à 50 000 USD/an` (= `<=50K`) | `income` |

Clique sur « Lancer l'analyse ». Le résultat s'affiche sur la droite. L'étiquette et le score proviennent **du vrai modèle**, pas d'une réponse simulée.

**Note métier :** la catégorie de revenu `<=50K` / `>50K` vient du recensement américain Adult, pas du contexte salarial tunisien. Ce résultat est expérimental et ne doit **jamais remplacer le diplôme déclaré/confirmé** par la personne.

## 4. Ce qui relie Django au modèle

```text
Navigateur / formulaire
         ↓
portal/urls.py → portal/views.py
         ↓ validation
portal/forms.py + portal/catalog.py
         ↓
ml_modules/education/adapter.py
         ↓
ml_modules/common.py (load_pipeline)
         ↓
ml_models/education/pipeline.joblib + model_metadata.json
         ↓
Résultat sur templates/portal/module.html
```

Le modèle attend **7 variables brutes** : `age`, `hours-per-week`, `capital-gain`, `capital-loss`, `workclass`, `occupation`, `income`.

Le pipeline `.joblib` contient **déjà** les étapes d'imputation, transformation log1p, encodage One-Hot et XGBoost. **Ne refais pas l'encodage dans la vue Django.** Le dossier `ml_models` doit rester privé : il ne doit pas être placé dans `static` ou exposé par une URL.

## 5. Tester l'API JSON

Endpoint : `POST /api/modules/education/predict/`

Exemple JSON attendu :

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

L'API utilise **la protection CSRF Django**. Pour un test simple via le navigateur, le formulaire HTML suffit. Pour une requête AJAX ou `curl`, récupère un jeton CSRF valide et envoie `X-CSRFToken` avec les cookies de la session. Ne supprime pas la protection CSRF juste pour le test. Les tests automatisés Django couvrent la route JSON.

## 6. Ajouter les cinq autres modèles

Chaque membre suit exactement le même contrat d'intégration :

1. Copier son modèle évalué dans `ml_models/<son_module>/pipeline.joblib`.
2. Adapter uniquement `ml_modules/<son_module>/adapter.py` pour recevoir les champs du formulaire et renvoyer `{'label': '...', 'note': '...'}` (plus `value` si régression).
3. Aligner la liste de variables dans **son** entrée de `portal/catalog.py`. Éviter toute fuite de la cible.
4. Écrire ou compléter des tests, puis faire sa Pull Request.

Les champs et formulaires des cinq autres modules sont **provisoires** : ils ne prouvent pas leur compatibilité avec les pipelines de tes coéquipiers. Sans modèle, Django affiche « module non connecté » plutôt qu'une prédiction inventée.

## 7. Mettre sur GitHub

Depuis ce dossier :

```bash
git init
git add .
git commit -m "feat: initialise application Django collaborative"
git branch -M main
git remote add origin https://github.com/VOTRE-UTILISATEUR/livrable_orentation.git
git push -u origin main
```

Si le dépôt existe déjà : **ne lance pas `git init` dans un dossier différent et ne force pas un `push`**. Mets les fichiers dans le dépôt cloné, puis crée une branche de fonctionnalité et une Pull Request.

**Important :** `.gitignore` ignore les artefacts `.joblib` et les CSV. Le ZIP que tu viens de recevoir **inclut** ton modèle Education pour les tests locaux, mais un `git add .` ne publiera pas le modèle. Après un clone depuis GitHub sur une autre machine, il faudra replacer ce modèle approuvé dans `ml_models/education/`. Tu peux partager l'artefact séparément via une archive privée ou un stockage d'équipe sécurisé.

## 8. Problèmes fréquents

| Problème | Solution |
|---|---|
| `python3.13: command not found` | Installer Python 3.13 puis créer le venv. |
| `No module named django` | Activer `.venv` et exécuter `pip install -r requirements.txt`. |
| `No module named xgboost` | Activer le même venv et installer les dépendances. |
| `libomp` manquant sur macOS | Installer OpenMP avec `brew install libomp`, puis relancer. |
| `FileNotFoundError: pipeline.joblib` | Vérifier le chemin `ml_models/education/pipeline.joblib`. |
| `Module non disponible` | Lancer `python scripts/doctor_model.py` et vérifier les 7 champs et les versions. |
| Erreur 403 sur l'API | Inclure cookie et token CSRF ; ne pas désactiver le CSRF. |
| Port 8000 déjà occupé | Lancer `python manage.py runserver 8001`. |
| Écran vide sans style | Vérifier que `STATICFILES_DIRS` pointe vers `static/`; recharger la page. |
| `ValueError` lors du chargement joblib | Utiliser Python et dépendances compatibles, ne jamais charger des artefacts non approuvés. |

## 9. Déploiement : après les tests locaux

Ne pas utiliser le serveur `runserver` en production. Avant une mise en ligne : secret Django via variable d'environnement, `DJANGO_DEBUG=0`, `ALLOWED_HOSTS`, serveur WSGI/ASGI, TLS/HTTPS, collecte des statiques, base adaptée, contrôle de débit, journaux sans données sensibles, et `python manage.py check --deploy`.

**Limites de l'évaluation :** ce modèle prédit un niveau à partir d'indicateurs socio-professionnels historiques américains. Il ne connaît pas directement le parcours scolaire confirmé d'un étudiant, et ses estimations ne constituent pas une admission ou une orientation définitive.

## 10. État des vérifications fournies

- ✅ Fichier `.joblib` inclus dans le ZIP, chargé avec les dépendances d'entraînement.
- ✅ Exemple de profil validé par le vrai pipeline (`0.5797`, seuil `0.52`).
- ✅ Syntaxe Python et arborescence contrôlées avant livraison.
- ⚠️ Le serveur et les tests Django complets doivent être exécutés dans un environnement où Django est installé. Environnement de génération sans accès réseau à Django.
