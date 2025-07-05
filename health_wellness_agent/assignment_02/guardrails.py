import re
from typing import List, Dict, Any

class SafetyGuardrails:
    def __init__(self):
        self.unsafe_patterns = [
            r"suicide|kill myself|end my life",
            r"self harm|cut myself|hurt myself",
            r"eating disorder|anorexia|bulimia",
            r"illegal drugs|steroids|controlled substances"
        ]
        
        self.medical_disclaimer = """
        ⚠️ Medical Disclaimer: This is AI-generated information for educational purposes only. 
        Always consult healthcare professionals for medical advice, diagnosis, or treatment.
        """
    
    def is_safe_input(self, text: str) -> bool:
        """Check if input is safe to process"""
        text_lower = text.lower()
        
        for pattern in self.unsafe_patterns:
            if re.search(pattern, text_lower):
                return False
        
        return True
    
    def add_medical_disclaimer(self, response: str) -> str:
        """Add medical disclaimer to response"""
        return f"{response}\n\n{self.medical_disclaimer}"
    
    def filter_response(self, response: str) -> str:
        """Filter and validate response"""
        # Add medical disclaimer for health-related responses
        if any(word in response.lower() for word in ["health", "medical", "doctor", "treatment", "symptom"]):
            response = self.add_medical_disclaimer(response)
        
        return response


