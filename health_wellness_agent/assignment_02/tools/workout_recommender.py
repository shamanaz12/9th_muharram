import google.generativeai as genai
from typing import Dict, Any, List

class WorkoutRecommender:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        genai.configure(api_key=config["api_key"])
        self.model = genai.GenerativeModel(config["model"])
    
    def recommend_workout(self, user_profile: Dict, goals: List, fitness_level: str) -> Dict[str, Any]:
        """Recommend workout routine"""
        try:
            prompt = f"""
            Create a workout routine for:
            
            Profile: {user_profile}
            Goals: {goals}
            Fitness Level: {fitness_level}
            
            Provide:
            - Weekly workout schedule
            - Exercise descriptions
            - Sets, reps, and rest periods
            - Progression plan
            - Safety tips
            - Equipment needed
            """
            
            response = self.model.generate_content(prompt)
            
            return {
                "workout_plan": response.text,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }
    
    def create_exercise_library(self, body_parts: List[str], equipment: List[str] = None) -> Dict[str, Any]:
        """Create exercise library for specific body parts"""
        try:
            prompt = f"""
            Create an exercise library for:
            Body Parts: {body_parts}
            Available Equipment: {equipment or 'Bodyweight only'}
            
            For each exercise provide:
            - Exercise name
            - Target muscles
            - Step-by-step instructions
            - Difficulty level
            - Modifications for beginners/advanced
            """
            
            response = self.model.generate_content(prompt)
            
            return {
                "exercise_library": response.text,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }

