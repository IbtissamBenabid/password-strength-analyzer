import secrets
import string
import re
import numpy as np
from typing import List, Tuple
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PasswordGenerator:
    def __init__(self):
        self.common_words = self._load_common_words()
        self.special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        
    def _load_common_words(self) -> List[str]:
        """Load common words for passphrase generation."""
        try:
            # Create data directory if it doesn't exist
            data_dir = Path('data')
            data_dir.mkdir(exist_ok=True)
            
            # Path to common words file
            words_file = data_dir / 'common_words.json'
            
            # If file doesn't exist, create it with some common words
            if not words_file.exists():
                common_words = [
                    "correct", "horse", "battery", "staple",
                    "sunshine", "rainbow", "butterfly", "dolphin",
                    "mountain", "ocean", "forest", "desert",
                    "computer", "keyboard", "monitor", "mouse"
                ]
                with open(words_file, 'w') as f:
                    json.dump(common_words, f)
            else:
                with open(words_file, 'r') as f:
                    common_words = json.load(f)
            
            return common_words
            
        except Exception as e:
            logger.error(f"Error loading common words: {str(e)}")
            return []
    
    def calculate_entropy(self, password: str) -> float:
        """Calculate Shannon entropy of a password."""
        if not password:
            return 0.0
        
        try:
            # Count character frequencies
            freq = {}
            for char in password:
                freq[char] = freq.get(char, 0) + 1
            
            # Calculate entropy using numpy for better numerical stability
            probabilities = np.array(list(freq.values())) / len(password)
            entropy = -np.sum(probabilities * np.log2(probabilities))
            
            return float(entropy)
            
        except Exception as e:
            logger.error(f"Error calculating entropy: {str(e)}")
            return 0.0
    
    def generate_random_password(self, length: int = 12) -> Tuple[str, float]:
        """Generate a random password with specified length."""
        try:
            # Ensure minimum length of 8 characters
            length = max(8, length)
            
            # Define character sets
            lowercase = string.ascii_lowercase
            uppercase = string.ascii_uppercase
            digits = string.digits
            special = self.special_chars
            
            # Ensure at least one character from each set
            password = [
                secrets.choice(lowercase),
                secrets.choice(uppercase),
                secrets.choice(digits),
                secrets.choice(special)
            ]
            
            # Fill the rest with random characters
            all_chars = lowercase + uppercase + digits + special
            password.extend(secrets.choice(all_chars) for _ in range(length - 4))
            
            # Shuffle the password
            password_list = list(password)
            secrets.SystemRandom().shuffle(password_list)
            password = ''.join(password_list)
            
            # Calculate entropy
            entropy = self.calculate_entropy(password)
            
            logger.info(f"Generated random password with entropy: {entropy:.2f}")
            return password, entropy
            
        except Exception as e:
            logger.error(f"Error generating random password: {str(e)}")
            raise
    
    def generate_passphrase(self, num_words: int = 4) -> Tuple[str, float]:
        """Generate a memorable passphrase."""
        try:
            # Ensure minimum of 3 words
            num_words = max(3, num_words)
            
            # Select random words
            words = [secrets.choice(self.common_words) for _ in range(num_words)]
            
            # Add random numbers and special characters
            numbers = ''.join(secrets.choice(string.digits) for _ in range(2))
            special = secrets.choice(self.special_chars)
            
            # Combine elements
            passphrase = f"{words[0].capitalize()}{special}{words[1]}{numbers}{words[2].capitalize()}"
            
            # Calculate entropy
            entropy = self.calculate_entropy(passphrase)
            
            logger.info(f"Generated passphrase with entropy: {entropy:.2f}")
            return passphrase, entropy
            
        except Exception as e:
            logger.error(f"Error generating passphrase: {str(e)}")
            raise
    
    def meets_nist_requirements(self, password: str) -> bool:
        """Check if password meets NIST requirements."""
        try:
            # Minimum length of 8 characters
            if len(password) < 8:
                return False
            
            # Check for character types
            has_lower = bool(re.search(r'[a-z]', password))
            has_upper = bool(re.search(r'[A-Z]', password))
            has_digit = bool(re.search(r'\d', password))
            has_special = bool(re.search(f'[{re.escape(self.special_chars)}]', password))
            
            # Must have at least 3 character types
            char_types = sum([has_lower, has_upper, has_digit, has_special])
            if char_types < 3:
                return False
            
            # Check for common patterns
            common_patterns = [
                r'123456',
                r'password',
                r'qwerty',
                r'admin',
                r'welcome'
            ]
            
            if any(re.search(pattern, password.lower()) for pattern in common_patterns):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking NIST requirements: {str(e)}")
            return False
    
    def estimate_cracking_time(self, entropy: float) -> str:
        """Estimate password cracking time based on entropy."""
        try:
            # Assuming 1 billion guesses per second
            guesses_per_second = 1e9
            
            # Calculate time in seconds
            seconds = 2 ** entropy / guesses_per_second
            
            # Convert to human-readable format
            if seconds < 60:
                return f"{seconds:.1f} seconds"
            elif seconds < 3600:
                return f"{seconds/60:.1f} minutes"
            elif seconds < 86400:
                return f"{seconds/3600:.1f} hours"
            elif seconds < 31536000:
                return f"{seconds/86400:.1f} days"
            else:
                return f"{seconds/31536000:.1f} years"
            
        except Exception as e:
            logger.error(f"Error estimating cracking time: {str(e)}")
            return "Unknown"

def main():
    """Main function to demonstrate usage."""
    generator = PasswordGenerator()
    
    # Generate and test random password
    password, entropy = generator.generate_random_password()
    nist_compliant = generator.meets_nist_requirements(password)
    cracking_time = generator.estimate_cracking_time(entropy)
    
    print(f"\nRandom Password:")
    print(f"Password: {password}")
    print(f"Entropy: {entropy:.2f} bits")
    print(f"NIST Compliant: {nist_compliant}")
    print(f"Estimated Cracking Time: {cracking_time}")
    
    # Generate and test passphrase
    passphrase, entropy = generator.generate_passphrase()
    nist_compliant = generator.meets_nist_requirements(passphrase)
    cracking_time = generator.estimate_cracking_time(entropy)
    
    print(f"\nPassphrase:")
    print(f"Passphrase: {passphrase}")
    print(f"Entropy: {entropy:.2f} bits")
    print(f"NIST Compliant: {nist_compliant}")
    print(f"Estimated Cracking Time: {cracking_time}")

if __name__ == "__main__":
    main() 