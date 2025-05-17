import unittest
import sys
from pathlib import Path

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent / 'src')
sys.path.append(src_path)

from password_generator import PasswordGenerator
from data_processing import PasswordProcessor

class TestPasswordSystem(unittest.TestCase):
    def setUp(self):
        self.generator = PasswordGenerator()
        self.processor = PasswordProcessor()
    
    def test_password_generation(self):
        """Test random password generation."""
        password, entropy = self.generator.generate_random_password()
        
        # Check password length
        self.assertGreaterEqual(len(password), 8)
        
        # Check character types
        self.assertTrue(any(c.islower() for c in password))
        self.assertTrue(any(c.isupper() for c in password))
        self.assertTrue(any(c.isdigit() for c in password))
        self.assertTrue(any(not c.isalnum() for c in password))
        
        # Check entropy
        self.assertGreaterEqual(entropy, 45)
    
    def test_passphrase_generation(self):
        """Test passphrase generation."""
        passphrase, entropy = self.generator.generate_passphrase()
        
        # Check passphrase length
        self.assertGreaterEqual(len(passphrase), 12)
        
        # Check entropy
        self.assertGreaterEqual(entropy, 45)
    
    def test_nist_compliance(self):
        """Test NIST compliance checking."""
        # Test weak password
        weak_password = "password123"
        self.assertFalse(self.generator.meets_nist_requirements(weak_password))
        
        # Test strong password
        strong_password = "X9m$kP2vN#qL"
        self.assertTrue(self.generator.meets_nist_requirements(strong_password))
    
    def test_feature_extraction(self):
        """Test password feature extraction."""
        password = "Test123!"
        features = self.processor.extract_features(password)
        
        # Check feature types
        self.assertIn('length', features)
        self.assertIn('lowercase', features)
        self.assertIn('uppercase', features)
        self.assertIn('digits', features)
        self.assertIn('special', features)
        self.assertIn('entropy', features)
        self.assertIn('has_common_pattern', features)
        
        # Check feature values
        self.assertEqual(features['length'], 8)
        self.assertEqual(features['lowercase'], 3)
        self.assertEqual(features['uppercase'], 1)
        self.assertEqual(features['digits'], 3)
        self.assertEqual(features['special'], 1)
    
    def test_entropy_calculation(self):
        """Test entropy calculation."""
        # Test empty password
        self.assertEqual(self.processor.calculate_entropy(""), 0.0)
        
        # Test simple password
        simple_entropy = self.processor.calculate_entropy("password")
        self.assertGreater(simple_entropy, 0)
        
        # Test complex password
        complex_entropy = self.processor.calculate_entropy("X9m$kP2vN#qL")
        self.assertGreater(complex_entropy, simple_entropy)
    
    def test_cracking_time_estimation(self):
        """Test cracking time estimation."""
        # Test low entropy
        time_low = self.generator.estimate_cracking_time(20)
        self.assertIn("seconds", time_low.lower())
        
        # Test high entropy
        time_high = self.generator.estimate_cracking_time(100)
        self.assertIn("years", time_high.lower())

if __name__ == '__main__':
    unittest.main() 