"""Branchement adapté au pipeline Education 2 classes préparé dans le projet précédent.

Le pipeline doit attendre les 7 variables BRUTES ci-dessous et contenir l'encodage.
Le seuil 0.52 a été choisi sur validation dans l'expérience précédente.
Il doit être revérifié si le modèle ou la population change.
"""
from ml_modules.common import load_pipeline, sample_frame, ModelNotReady

MAPPING = {'hours_per_week': 'hours-per-week', 'capital_gain': 'capital-gain', 'capital_loss': 'capital-loss'}
THRESHOLD = 0.52
POSITIVE = 'Niveau supérieur'
NEGATIVE = 'Niveau non supérieur'


def predict(profile):
    model = load_pipeline('education')
    x = sample_frame(profile, MAPPING)
    if not hasattr(model, 'predict_proba'):
        raise ModelNotReady('Le pipeline education doit fournir predict_proba().')
    try:
        classes = [str(c) for c in model.classes_]
        # Le pipeline existant a les labels numeriques [0, 1] (1 = superieur).
        # Un pipeline alternatif peut avoir des labels textuels.
        if '1' in classes:
            idx = classes.index('1')
        else:
            idx = classes.index(POSITIVE)
        score = float(model.predict_proba(x)[0, idx])
    except (ValueError, IndexError, AttributeError) as exc:
        raise ModelNotReady('Classes du pipeline incompatibles : vérifier model.classes_.') from exc
    except Exception as exc:
        raise ModelNotReady('Features du pipeline incompatibles : vérifier les 7 champs Adult dans adapter.py.') from exc
    label = POSITIVE if score >= THRESHOLD else NEGATIVE
    return {'label': label, 'probability_superior': round(score, 4),
            'threshold': THRESHOLD,
            'note': 'Ce score ne reflète pas un diplôme vérifié. Demandez toujours confirmation.'}
