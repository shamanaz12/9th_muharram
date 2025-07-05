import google.generativeai as genai
from typing import Dict, Any, List

class MealPlanner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        genai.configure(api_key=config["api_key"])
        self.model = genai.GenerativeModel(config["model"])
    
    def create_meal_plan(self, user_profile: Dict, goals: List, preferences: Dict = None) -> Dict[str, Any]:
        """Create personalized meal plan"""
        try:
            prompt = f"""
            Create a personalized meal plan:
            
            User Profile: {user_profile}
            Goals: {goals}
            Preferences: {preferences or 'None specified'}
            
            Provide a 7-day meal plan with:
            - Breakfast, lunch, dinner, and snacks
            - Nutritional information
            - Shopping list
            - Preparation tips
            - Portion sizes
            """
            
            response = self.model.generate_content(prompt)
            
            return {
                "meal_plan": response.text,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }
    
    def suggest_healthy_recipes(self, dietary_requirements: List[str]) -> Dict[str, Any]:
        """Suggest healthy recipes based on dietary requirements"""
        try:
            prompt = f"""
            Suggest 5 healthy recipes for someone with these dietary requirements:
            {dietary_requirements}
            
            For each recipe provide:
            - Recipe name
            - Ingredients list
            - Cooking instructions
            - Nutritional benefits
            - Prep time and cook time
            """
            
            response = self.model.generate_content(prompt)
            
            return {
                "recipes": response.text,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }