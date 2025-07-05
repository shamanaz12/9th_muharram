import google.generativeai as genai
from typing import Dict, Any, List

class NutritionExpertAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        genai.configure(api_key=config["api_key"])
        self.model = genai.GenerativeModel(config["model"])
    
    def analyze_nutrition(self, food_items: List[str], user_goals: List[str] = None) -> Dict[str, Any]:
        """Analyze nutritional content of food items"""
        try:
            prompt = f"""
            As a nutrition expert, analyze these food items:
            Foods: {food_items}
            User Goals: {user_goals or 'General health'}
            
            Provide:
            1. Nutritional breakdown (calories, macros, vitamins, minerals)
            2. Health benefits
            3. Potential concerns
            4. Recommendations for improvement
            5. Portion size suggestions
            6. Meal timing advice
            """
            
            response = self.model.generate_content(prompt)
            
            return {
                "nutrition_analysis": response.text,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }
    
    def create_dietary_recommendations(self, health_conditions: List[str], preferences: Dict) -> Dict[str, Any]:
        """Create dietary recommendations based on health conditions"""
        try:
            prompt = f"""
            Create dietary recommendations for someone with:
            Health Conditions: {health_conditions}
            Preferences: {preferences}
            
            Provide:
            1. Foods to emphasize
            2. Foods to limit or avoid
            3. Meal timing strategies
            4. Supplement recommendations
            5. Hydration guidelines
            6. Special considerations
            """
            
            response = self.model.generate_content(prompt)
            
            return {
                "dietary_recommendations": response.text,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }