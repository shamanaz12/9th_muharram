import google.generativeai as genai
from typing import Dict, Any, List
import json

class HealthWellnessAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_genai()
        
    def setup_genai(self):
        """Configure Gemini API"""
        genai.configure(api_key=self.config["api_key"])
        self.model = genai.GenerativeModel(self.config["model"])
        
    def process_query(self, query: str, context: Dict = None) -> str:
        """Process user query with context"""
        try:
            system_prompt = self._get_system_prompt()
            
            # Format prompt with context
            full_prompt = self._format_prompt(query, context, system_prompt)
            
            # Generate response
            response = self.model.generate_content(full_prompt)
            
            return response.text
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for health wellness agent"""
        return """
        You are a Health & Wellness AI Assistant. Your role is to:
        
        1. Provide helpful, accurate health and wellness information
        2. Create personalized fitness and nutrition plans
        3. Track user progress and goals
        4. Offer motivation and support
        5. Escalate serious health concerns to professionals
        
        Guidelines:
        - Always prioritize user safety
        - Provide evidence-based advice
        - Be empathetic and supportive
        - Encourage professional consultation when needed
        - Never diagnose medical conditions
        
        Respond in a friendly, professional manner.
        """
    
    def _format_prompt(self, query: str, context: Dict, system_prompt: str) -> str:
        """Format the complete prompt"""
        context_str = ""
        if context:
            context_str = f"\nUser Context: {json.dumps(context, indent=2)}\n"
        
        return f"{system_prompt}\n{context_str}\nUser Query: {query}\n\nResponse:"

