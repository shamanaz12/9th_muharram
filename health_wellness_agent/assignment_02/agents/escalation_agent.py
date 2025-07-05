import google.generativeai as genai
from typing import Dict, Any, List

class EscalationAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        genai.configure(api_key=config["api_key"])
        self.model = genai.GenerativeModel(config["model"])
        
        self.escalation_triggers = [
            "chest pain", "difficulty breathing", "severe pain",
            "blood", "injury", "emergency", "urgent", "serious symptoms"
        ]
    
    def should_escalate(self, query: str) -> bool:
        """Check if query should be escalated"""
        query_lower = query.lower()
        return any(trigger in query_lower for trigger in self.escalation_triggers)
    
    def handle_escalation(self, query: str, context: Dict = None) -> Dict[str, Any]:
        """Handle escalated queries"""
        try:
            prompt = f"""
            This query requires professional medical attention:
            Query: {query}
            Context: {context or 'Not provided'}
            
            Provide:
            1. Immediate advice (if safe to give)
            2. Urgency level (low, medium, high, emergency)
            3. When to seek help (now, today, this week)
            4. What type of professional to contact
            5. What information to prepare for the consultation
            
            Always emphasize seeking professional help.
            """
            
            response = self.model.generate_content(prompt)
            
            return {
                "escalation_response": response.text,
                "requires_immediate_attention": True,
                "status": "escalated"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }
