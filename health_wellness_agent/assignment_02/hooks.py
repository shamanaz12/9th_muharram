from typing import Callable, Any
from context import ContextManager
from guardrails import SafetyGuardrails

def setup_hooks(context_manager: ContextManager, safety_guardrails: SafetyGuardrails):
    """Setup hooks for the application"""
    
    def pre_process_hook(query: str, context: dict) -> tuple:
        """Hook called before processing query"""
        # Log interaction
        print(f"🔍 Processing query: {query[:50]}...")
        
        # Safety check
        if not safety_guardrails.is_safe_input(query):
            raise ValueError("Unsafe input detected")
        
        return query, context
    
    def post_process_hook(query: str, response: str, context: dict) -> str:
        """Hook called after processing query"""
        # Add interaction to context
        context_manager.add_interaction(query, response)
        
        # Apply safety filters
        filtered_response = safety_guardrails.filter_response(response)
        
        print(f"✅ Response generated successfully")
        
        return filtered_response
    
    return pre_process_hook, post_process_hook