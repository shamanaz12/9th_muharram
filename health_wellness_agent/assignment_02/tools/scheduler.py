from datetime import datetime, timedelta
from typing import Dict, Any, List

class Scheduler:
    def __init__(self):
        self.schedule = {}
        self.reminders = []
    
    def create_workout_schedule(self, workout_plan: Dict, user_preferences: Dict) -> Dict[str, Any]:
        """Create workout schedule"""
        try:
            # Extract preferred times and days
            preferred_days = user_preferences.get("workout_days", ["Monday", "Wednesday", "Friday"])
            preferred_time = user_preferences.get("workout_time", "morning")
            
            schedule = {}
            for day in preferred_days:
                schedule[day] = {
                    "workout_type": "Full body workout",  # This would be determined from workout_plan
                    "time": preferred_time,
                    "duration": "45 minutes",
                    "exercises": []  # Would be populated from workout_plan
                }
            
            return {
                "schedule": schedule,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }
    
    def set_reminder(self, reminder_type: str, time: str, message: str) -> Dict[str, Any]:
        """Set reminder for user"""
        try:
            reminder = {
                "id": len(self.reminders) + 1,
                "type": reminder_type,
                "time": time,
                "message": message,
                "created_at": datetime.now().isoformat(),
                "active": True
            }
            
            self.reminders.append(reminder)
            
            return {
                "reminder": reminder,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }
    
    def get_daily_schedule(self, date: str = None) -> Dict[str, Any]:
        """Get schedule for specific date"""
        if not date:
            date = datetime.now().strftime("%A")
        
        return {
            "date": date,
            "schedule": self.schedule.get(date, {}),
            "reminders": [r for r in self.reminders if r["active"]]
        }
