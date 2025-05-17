import pytest
from password_analysis import extract_features, calculate_entropy

class TestPasswordAnalysis:
    def test_extract_features_basic(self):
        """Test basic feature extraction"""
        password = "Test123!"
        features = extract_features(password)
        
        assert features['length'] == 8
        assert features['lowercase'] == 3
        assert features['uppercase'] == 1
        assert features['digits'] == 3
        assert features['special'] == 1
        assert isinstance(features['entropy'], float)

    def test_extract_features_empty(self):
        """Test feature extraction with empty password"""
        password = ""
        features = extract_features(password)
        
        assert features['length'] == 0
        assert features['lowercase'] == 0
        assert features['uppercase'] == 0
        assert features['digits'] == 0
        assert features['special'] == 0
        assert features['entropy'] == 0.0

    def test_extract_features_special_chars(self):
        """Test feature extraction with special characters"""
        password = "!@#$%^&*()"
        features = extract_features(password)
        
        assert features['length'] == 10
        assert features['special'] == 10
        assert features['entropy'] > 0

    def test_calculate_entropy(self):
        """Test entropy calculation"""
        # Test with a simple password
        assert calculate_entropy("password") > 0
        
        # Test with a complex password
        complex_pass = "P@ssw0rd!123"
        assert calculate_entropy(complex_pass) > calculate_entropy("password")
        
        # Test with empty password
        assert calculate_entropy("") == 0.0

    @pytest.mark.parametrize("password,expected_length", [
        ("abc", 3),
        ("ABC123", 6),
        ("!@#$%^", 6),
        ("", 0),
    ])
    def test_password_length(self, password, expected_length):
        """Test password length calculation with various inputs"""
        features = extract_features(password)
        assert features['length'] == expected_length

    def test_feature_consistency(self):
        """Test that features are consistent with password content"""
        password = "Test123!@#"
        features = extract_features(password)
        
        # Sum of all character types should equal length
        total_chars = (features['lowercase'] + 
                      features['uppercase'] + 
                      features['digits'] + 
                      features['special'])
        assert total_chars == features['length'] 