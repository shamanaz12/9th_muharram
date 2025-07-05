from typing import Dict, Any, List
from datetime import datetime

class ContextManager:
    def __init__(self):
        self.user_context = {
            "profile": {},
            "goals": [],
            "preferences": {},
            "history": [],
            "current_session": {
                "start_time": datetime.now().isoformat(),
                "interactions": []
            }
        }
    
    def update_context(self, new_context: Dict[str, Any]):
        """Update user context with new information"""
        if "profile" in new_context:
            self.user_context["profile"].update(new_context["profile"])
        
        if "goals" in new_context:
            self.user_context["goals"].extend(new_context["goals"])
        
        if "preferences" in new_context:
            self.user_context["preferences"].update(new_context["preferences"])
    
    def add_interaction(self, query: str, response: str):
        """Add interaction to current session"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response
        }
        self.user_context["current_session"]["interactions"].append(interaction)
    
    def get_context(self) -> Dict[str, Any]:
        """Get current user context"""
        return self.user_context
    
    def get_profile(self) -> Dict[str, Any]:
        """Get user profile"""
        return self.user_context.get("profile", {})
    
    def get_goals(self) -> List[Dict[str, Any]]:
        """Get user goals"""
        return self.user_context.get("goals", [])
