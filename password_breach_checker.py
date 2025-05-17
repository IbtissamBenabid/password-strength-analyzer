import hashlib
import requests
from typing import Dict, Optional, Tuple
import time

class PasswordBreachChecker:
    """Class to check passwords against the HaveIBeenPwned API."""
    
    def __init__(self):
        self.api_url = "https://api.pwnedpasswords.com/range/"
        self.headers = {
            "User-Agent": "PasswordStrengthAnalyzer",
            "Add-Padding": "true"
        }
    
    def _hash_password(self, password: str) -> str:
        """Hash the password using SHA-1."""
        return hashlib.sha1(password.encode('utf-8')).hexdigest().upper()
    
    def _get_hash_prefix_suffix(self, hashed_password: str) -> Tuple[str, str]:
        """Split the hash into prefix and suffix."""
        return hashed_password[:5], hashed_password[5:]
    
    def check_password(self, password: str) -> Tuple[bool, int, Optional[str]]:
        """
        Check if a password has been breached.
        
        Args:
            password (str): The password to check
            
        Returns:
            Tuple[bool, int, Optional[str]]: 
                - Whether the password was found in breaches
                - Number of times the password was found
                - Error message if any
        """
        try:
            # Hash the password
            hashed_password = self._hash_password(password)
            prefix, suffix = self._get_hash_prefix_suffix(hashed_password)
            
            # Make API request
            response = requests.get(
                f"{self.api_url}{prefix}",
                headers=self.headers,
                timeout=5
            )
            
            if response.status_code != 200:
                return False, 0, f"API request failed with status code: {response.status_code}"
            
            # Parse response
            hashes = (line.split(':') for line in response.text.splitlines())
            for hash_suffix, count in hashes:
                if hash_suffix == suffix:
                    return True, int(count), None
            
            return False, 0, None
            
        except requests.exceptions.RequestException as e:
            return False, 0, f"Network error: {str(e)}"
        except Exception as e:
            return False, 0, f"Error checking password: {str(e)}"
    
    def get_breach_details(self, password: str) -> Dict:
        """
        Get detailed breach information for a password.
        
        Args:
            password (str): The password to check
            
        Returns:
            Dict: Dictionary containing breach information
        """
        is_breached, count, error = self.check_password(password)
        
        if error:
            return {
                "is_breached": False,
                "count": 0,
                "error": error,
                "risk_level": "unknown",
                "message": f"Unable to check password: {error}"
            }
        
        if is_breached:
            risk_level = "high" if count > 1000 else "medium" if count > 100 else "low"
            message = (
                f"This password has been found {count:,} times in data breaches. "
                f"It is recommended to use a different password."
            )
        else:
            risk_level = "safe"
            message = "This password has not been found in any known data breaches."
        
        return {
            "is_breached": is_breached,
            "count": count,
            "error": None,
            "risk_level": risk_level,
            "message": message
        } 