import pytest
import joblib
import os
from password_analysis import extract_features
from password_strength_ui import load_model, generate_password_by_criteria

class TestIntegration:
    @pytest.fixture
    def model(self):
        """Load the trained model for testing"""
        model = load_model()
        assert model is not None, "Model should be loaded successfully"
        return model

    def test_model_prediction(self, model):
        """Test end-to-end password strength prediction"""
        # Test with a weak password
        weak_pass = "password123"
        features = extract_features(weak_pass)
        prediction = model.predict([list(features.values())])[0]
        assert prediction in ['very_weak', 'weak', 'average', 'strong', 'very_strong']

        # Test with a strong password
        strong_pass = "P@ssw0rd!2023#Secure"
        features = extract_features(strong_pass)
        prediction = model.predict([list(features.values())])[0]
        assert prediction in ['strong', 'very_strong']

    def test_password_generation(self):
        """Test password generation with different criteria"""
        # Test basic criteria
        criteria = {
            'length': 12,
            'lowercase': True,
            'uppercase': True,
            'digits': True,
            'special': True,
            'exclude_similar': False,
            'exclude_ambiguous': False,
            'require_unique': False
        }
        
        password = generate_password_by_criteria(criteria)
        assert password is not None
        assert len(password) == 12
        
        # Verify password meets criteria
        features = extract_features(password)
        assert features['lowercase'] > 0
        assert features['uppercase'] > 0
        assert features['digits'] > 0
        assert features['special'] > 0

    def test_password_generation_unique(self):
        """Test password generation with unique characters requirement"""
        criteria = {
            'length': 8,
            'lowercase': True,
            'uppercase': True,
            'digits': True,
            'special': True,
            'exclude_similar': False,
            'exclude_ambiguous': False,
            'require_unique': True
        }
        
        password = generate_password_by_criteria(criteria)
        assert password is not None
        assert len(set(password)) == len(password)

    def test_model_metrics(self):
        """Test model metrics loading and validation"""
        metrics = load_model_metrics()
        assert metrics is not None, "Model metrics should be loaded successfully"
        
        # Verify required metrics are present
        required_metrics = ['accuracy', 'precision', 'recall', 'confusion_matrix']
        for metric in required_metrics:
            assert metric in metrics, f"Missing required metric: {metric}"

    def test_feature_extraction_pipeline(self, model):
        """Test the complete feature extraction and prediction pipeline"""
        test_passwords = [
            "password123",
            "P@ssw0rd!2023",
            "abc123",
            "!@#$%^&*()",
            ""
        ]
        
        for password in test_passwords:
            # Extract features
            features = extract_features(password)
            assert isinstance(features, dict)
            
            # Make prediction
            prediction = model.predict([list(features.values())])[0]
            assert prediction in ['very_weak', 'weak', 'average', 'strong', 'very_strong']
            
            # Get probabilities
            probabilities = model.predict_proba([list(features.values())])[0]
            assert len(probabilities) == 5  # Five strength levels
            assert sum(probabilities) == pytest.approx(1.0)  # Probabilities should sum to 1 