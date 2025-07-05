import google.generativeai as genai
from typing import Dict, Any, List

class GoalAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        genai.configure(api_key=config["api_key"])
        self.model = genai.GenerativeModel(config["model"])
    
    def analyze_goals(self, user_goals: str, user_profile: Dict = None) -> Dict[str, Any]:
        """Analyze user goals and create action plan"""
        try:
            prompt = f"""
            Analyze these health/fitness goals and create a structured plan:
            
            Goals: {user_goals}
            Profile: {user_profile or 'Not provided'}
            
            Provide analysis in JSON format with:
            - goal_type: (weight_loss, muscle_gain, endurance, general_health)
            - timeline: estimated time to achieve
            - difficulty_level: (beginner, intermediate, advanced)
            - key_milestones: list of milestones
            - recommended_approach: detailed approach
            - potential_challenges: list of challenges
            """
            
            response = self.model.generate_content(prompt)
            
            # Parse response (in real implementation, you'd parse JSON)
            return {
                "analysis": response.text,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }
    
    def track_progress(self, goals: List[Dict], current_status: Dict) -> Dict:
        """Track progress towards goals"""
        try:
            prompt = f"""
            Track progress for these goals:
            Goals: {goals}
            Current Status: {current_status}
            
            Provide progress update with:
            - completion_percentage
            - achievements
            - areas_for_improvement
            - next_steps
            """
            
            response = self.model.generate_content(prompt)
            
            return {
                "progress_report": response.text,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }
