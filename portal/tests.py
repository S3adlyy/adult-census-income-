import json
from unittest.mock import patch
from django.test import TestCase, Client
from django.urls import reverse
from .catalog import MODULES, BY_SLUG
from .forms import create_profile_form
from ml_modules.common import ModelNotReady

class PortalSmokeTests(TestCase):
    def test_homepage_loads_and_shows_six_modules(self):
        response = self.client.get(reverse('portal:home'))
        self.assertEqual(response.status_code, 200)
        for module in MODULES:
            self.assertContains(response, module.title)

    def test_each_module_page(self):
        for module in MODULES:
            with self.subTest(module=module.slug):
                response = self.client.get(reverse('portal:module', args=[module.slug]))
                self.assertEqual(response.status_code, 200)
                self.assertContains(response, module.title)

    def test_education_features_exclude_leakage(self):
        keys = {field.key for field in BY_SLUG['education'].fields}
        self.assertEqual(len(keys), 7)
        self.assertNotIn('education', keys)
        self.assertNotIn('education_num', keys)

    def test_other_modules_exclude_target(self):
        for module in MODULES:
            keys = {f.key for f in module.fields}
            self.assertNotIn(module.target, keys)
            if module.slug == 'age': self.assertNotIn('age', keys)
            if module.slug == 'hours': self.assertNotIn('hours_per_week', keys)
            if module.slug == 'workclass': self.assertNotIn('workclass', keys)
            if module.slug == 'income': self.assertNotIn('income', keys)
            if module.slug == 'occupation': self.assertNotIn('occupation', keys)

    def test_invalid_profile_does_not_call_model(self):
        with patch('portal.views.execute') as execute:
            res = self.client.post(reverse('portal:module', args=['education']), {'age': -4})
            self.assertEqual(res.status_code, 200)
            execute.assert_not_called()

    def test_unknown_module_is_404(self):
        self.assertEqual(self.client.get('/modules/unknown/').status_code, 404)

    def test_health(self):
        res = self.client.get(reverse('portal:health'))
        self.assertEqual(res.json()['modules'], 6)

    def test_api_invalid_json(self):
        url = reverse('portal:predict_api', args=['education'])
        res = self.client.post(url, data='{', content_type='application/json')
        self.assertEqual(res.status_code, 400)

    def test_csrf_protects_api(self):
        client = Client(enforce_csrf_checks=True)
        url = reverse('portal:predict_api', args=['education'])
        res = client.post(url, data='{}', content_type='application/json')
        self.assertEqual(res.status_code, 403)

    def test_api_without_model_is_503(self):
        url = reverse('portal:predict_api', args=['education'])
        payload = {'age':28,'hours_per_week':40,'capital_gain':0,'capital_loss':0,
                   'workclass':'Private','occupation':'Adm-clerical','income':'<=50K'}
        with patch('portal.views.execute', side_effect=ModelNotReady('Pas de modèle')):
            res = self.client.post(url, data=json.dumps(payload), content_type='application/json')
        self.assertEqual(res.status_code, 503)
        self.assertEqual(res.json()['status'], 'not_ready')

    def test_api_successful_adapter_contract(self):
        url = reverse('portal:predict_api', args=['education'])
        payload = {'age':28,'hours_per_week':40,'capital_gain':0,'capital_loss':0,
                   'workclass':'Private','occupation':'Adm-clerical','income':'<=50K'}
        with patch('portal.views.execute', return_value={'label': 'Niveau supérieur', 'probability_superior': 0.6}):
            res = self.client.post(url, data=json.dumps(payload), content_type='application/json')
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()['result']['label'], 'Niveau supérieur')
