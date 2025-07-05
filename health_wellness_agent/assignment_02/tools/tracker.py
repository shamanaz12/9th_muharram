from datetime import datetime
from typing import Dict, Any, List

class ProgressTracker:
    def __init__(self):
        self.progress_data = {
            "weight": [],
            "workouts": [],
            "meals": [],
            "measurements": [],
            "achievements": []
        }
    
    def log_workout(self, workout_data: Dict[str, Any]) -> Dict[str, Any]:
        """Log workout session"""
        try:
            workout_entry = {
                "date": datetime.now().isoformat(),
                "type": workout_data.get("type", "General"),
                "duration": workout_data.get("duration", 0),
                "exercises": workout_data.get("exercises", []),
                "calories_burned": workout_data.get("calories_burned", 0),
                "intensity": workout_data.get("intensity", "moderate")
            }
            
            self.progress_data["workouts"].append(workout_entry)
            
            return {
                "logged_workout": workout_entry,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }
    
    def log_weight(self, weight: float, unit: str = "kg") -> Dict[str, Any]:
        """Log weight measurement"""
        try:
            weight_entry = {
                "date": datetime.now().isoformat(),
                "weight": weight,
                "unit": unit
            }
            
            self.progress_data["weight"].append(weight_entry)
            
            return {
                "logged_weight": weight_entry,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }
    
    def get_progress_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get progress summary for specified days"""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Filter data by date range
            recent_workouts = [
                w for w in self.progress_data["workouts"]
                if datetime.fromisoformat(w["date"]) >= start_date
            ]
            
            recent_weights = [
                w for w in self.progress_data["weight"]
                if datetime.fromisoformat(w["date"]) >= start_date
            ]
            
            summary = {
                "period": f"Last {days} days",
                "workouts_completed": len(recent_workouts),
                "weight_entries": len(recent_weights),
                "achievements": self.progress_data["achievements"][-5:],  # Last 5 achievements
                "trends": self._calculate_trends(recent_weights, recent_workouts)
            }
            
            return {
                "summary": summary,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }
    
    def _calculate_trends(self, weights: List[Dict], workouts: List[Dict]) -> Dict[str, Any]:
        """Calculate trends from data"""
        trends = {}
        
        if len(weights) >= 2:
            latest_weight = weights[-1]["weight"]
            earliest_weight = weights[0]["weight"]
            weight_change = latest_weight - earliest_weight
            trends["weight_change"] = f"{weight_change:+.1f} kg"
        
        if workouts:
            avg_workouts_per_week = len(workouts) / 4  # Assuming 4 weeks
            trends["workout_frequency"] = f"{avg_workouts_per_week:.1f} workouts/week"
        
        return trends