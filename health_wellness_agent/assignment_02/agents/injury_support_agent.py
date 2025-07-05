import google.generativeai as genai
from typing import Dict, Any, List

class InjurySupportAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        genai.configure(api_key=config["api_key"])
        self.model = genai.GenerativeModel(config["model"])
    
    def assess_injury(self, injury_description: str, symptoms: List[str]) -> Dict[str, Any]:
        """Assess injury and provide initial guidance"""
        try:
            prompt = f"""
            Assess this injury situation:
            Description: {injury_description}
            Symptoms: {symptoms}
            
            Provide:
            1. Severity assessment (requires immediate attention or can wait)
            2. Initial care recommendations (RICE protocol, etc.)
            3. Warning signs to watch for
            4. When to seek professional help
            5. Recovery timeline estimates
            6. Prevention tips for future
            
            IMPORTANT: Always recommend professional evaluation for injuries.
            """
            
            response = self.model.generate_content(prompt)
            
            return {
                "injury_assessment": response.text,
                "seek_professional_help": True,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }
    
    def rehabilitation_guidance(self, injury_type: str, recovery_stage: str) -> Dict[str, Any]:
        """Provide rehabilitation guidance"""
        try:
            prompt = f"""
            Provide rehabilitation guidance for:
            Injury Type: {injury_type}
            Recovery Stage: {recovery_stage}
            
            Include:
            1. Appropriate exercises for current stage
            2. Activities to avoid
            3. Progression criteria
            4. Pain management strategies
            5. Nutrition for healing
            6. When to advance to next stage
            
            Emphasize working with healthcare professionals.
            """
            
            response = self.model.generate_content(prompt)
            
            return {
                "rehabilitation_guidance": response.text,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }
