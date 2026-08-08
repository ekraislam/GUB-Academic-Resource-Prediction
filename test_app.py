"""
test_app.py
-----------
Automated test script to verify all Flask routes and prediction flow.
"""

import app as flask_app
import unittest

class TestGUBApp(unittest.TestCase):
    def setUp(self):
        flask_app.app.config['TESTING'] = True
        flask_app.app.config['WTF_CSRF_ENABLED'] = False
        self.client = flask_app.app.test_client()

    def test_index_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Academic Resource', response.data)
        self.assertIn(b'Dashboard', response.data)

    def test_predict_page(self):
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Predict Resource Demand', response.data)

    def test_prediction_flow(self):
        # Test input 1: High demand scenario
        form_data_1 = {
            "department": "CSE",
            "course_code": "CSE401",
            "course_type": "Theory+Lab",
            "credit": "3",
            "semester": "Fall",
            "previous_enrollment": "180",
            "previous_sections": "3",
            "previous_pass_rate": "55",
            "course_popularity": "9",
            "lab_required": "1",
            "lab_capacity": "30",
            "room_capacity": "50",
            "instructor_count": "2",
            "class_duration": "90",
            "previous_resource_usage": "90",
            "registration_growth_rate": "30"
        }
        res1 = self.client.post('/predict', data=form_data_1, follow_redirects=True)
        self.assertEqual(res1.status_code, 200)
        self.assertIn(b'RESOURCE DEMAND', res1.data)
        self.assertIn(b'CSE', res1.data)

        # Test input 2: Low demand scenario
        form_data_2 = {
            "department": "English",
            "course_code": "ENG101",
            "course_type": "Theory",
            "credit": "3",
            "semester": "Spring",
            "previous_enrollment": "20",
            "previous_sections": "1",
            "previous_pass_rate": "90",
            "course_popularity": "2",
            "lab_required": "0",
            "lab_capacity": "0",
            "room_capacity": "60",
            "instructor_count": "1",
            "class_duration": "90",
            "previous_resource_usage": "25",
            "registration_growth_rate": "-10"
        }
        res2 = self.client.post('/predict', data=form_data_2, follow_redirects=True)
        self.assertEqual(res2.status_code, 200)
        self.assertIn(b'RESOURCE DEMAND', res2.data)

        # Test input 3: Medium demand scenario
        form_data_3 = {
            "department": "EEE",
            "course_code": "EEE201",
            "course_type": "Theory+Lab",
            "credit": "4",
            "semester": "Summer",
            "previous_enrollment": "75",
            "previous_sections": "2",
            "previous_pass_rate": "78",
            "course_popularity": "6",
            "lab_required": "1",
            "lab_capacity": "30",
            "room_capacity": "50",
            "instructor_count": "2",
            "class_duration": "90",
            "previous_resource_usage": "65",
            "registration_growth_rate": "8"
        }
        res3 = self.client.post('/predict', data=form_data_3, follow_redirects=True)
        self.assertEqual(res3.status_code, 200)
        self.assertIn(b'RESOURCE DEMAND', res3.data)

    def test_model_performance_page(self):
        response = self.client.get('/model-performance')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Model Performance', response.data)
        self.assertIn(b'Logistic Regression', response.data)

    def test_about_page(self):
        response = self.client.get('/about')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'GUB Academic AI', response.data)
        self.assertIn(b'Dataset Disclaimer', response.data)

    def test_plots_route(self):
        response = self.client.get('/plots/demand_distribution.png')
        self.assertEqual(response.status_code, 200)

    def test_export_report_route(self):
        form_data = {
            "department": "CSE",
            "course_code": "CSE401",
            "course_type": "Theory+Lab",
            "credit": "3",
            "semester": "Fall",
            "previous_enrollment": "180",
            "previous_sections": "3",
            "previous_pass_rate": "55",
            "course_popularity": "9",
            "lab_required": "1",
            "lab_capacity": "30",
            "room_capacity": "50",
            "instructor_count": "2",
            "class_duration": "90",
            "previous_resource_usage": "90",
            "registration_growth_rate": "30"
        }
        self.client.post('/predict', data=form_data, follow_redirects=True)
        response = self.client.get('/export-report')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'application/pdf')
        self.assertTrue(response.data.startswith(b'%PDF'))

if __name__ == '__main__':

    unittest.main()
