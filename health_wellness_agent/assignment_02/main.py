import chainlit as cl
import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import traceback
import asyncio # Added for running sync functions in async context

# Import your existing modules
from agent import HealthWellnessAgent
from context import ContextManager
from guardrails import SafetyGuardrails
from hooks import setup_hooks
from tools.goal_analyzer import GoalAnalyzer
from tools.meal_planner import MealPlanner
from tools.workout_recommender import WorkoutRecommender
from tools.scheduler import Scheduler
from tools.tracker import ProgressTracker
from agents.escalation_agent import EscalationAgent
from agents.nutrition_expert_agent import NutritionExpertAgent
from agents.injury_support_agent import InjurySupportAgent
from utils.streaming import StreamingChat


# Load environment variables
load_dotenv()

# --- Your existing HealthWellnessApp class ---
class HealthWellnessApp:
    def __init__(self):
        self.setup_configuration()
        self.initialize_components()
        self.setup_agents()
        self.setup_tools()
        
    def setup_configuration(self):
        """Initialize API configuration"""
        self.model_name = "gemini-1.5-flash"
        self.api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it.")
            
        try:
            genai.configure(api_key=self.api_key)
            print("Gemini API configured successfully in HealthWellnessApp.")
        except Exception as e:
            print(f"Warning: Could not configure genai directly: {e}. Ensure StreamingChat handles API key.")

        self.base_config = {
            "api_key": self.api_key,
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", 
            "model": self.model_name
        }
    
    def initialize_components(self):
        """Initialize core components"""
        self.context_manager = ContextManager()
        self.safety_guardrails = SafetyGuardrails()
        self.streaming_chat = StreamingChat(self.base_config) 
        
        setup_hooks(self.context_manager, self.safety_guardrails)
    
    def setup_agents(self):
        """Initialize all agents"""
        self.main_agent = HealthWellnessAgent(self.base_config)
        self.escalation_agent = EscalationAgent(self.base_config)
        self.nutrition_agent = NutritionExpertAgent(self.base_config)
        self.injury_agent = InjurySupportAgent(self.base_config)
    
    def setup_tools(self):
        """Initialize all tools"""
        self.goal_analyzer = GoalAnalyzer(self.base_config)
        self.meal_planner = MealPlanner(self.base_config)
        self.workout_recommender = WorkoutRecommender(self.base_config)
        self.scheduler = Scheduler()
        self.progress_tracker = ProgressTracker()
    
    async def process_user_input_async(self, user_input, user_context=None):
        """Main processing function adapted for async Chainlit environment"""
        try:
            if not self.safety_guardrails.is_safe_input(user_input):
                return "I can't help with that request. Please ask something related to health and wellness."
            
            if user_context:
                self.context_manager.update_context(user_context)
            
            # FIX: Use asyncio.to_thread for running the synchronous process_query
            # This is a more robust way to run sync code in an async context
            response = await asyncio.to_thread(
                self.main_agent.process_query, # The synchronous function to run
                user_input, # Arguments to the function
                self.context_manager.get_context()
            )
            
            return response
            
        except Exception as e:
            traceback.print_exc()
            return f"Error processing your request: {str(e)}. Please try again."

# --- Chainlit Integration ---

health_app_instance = None

@cl.on_chat_start
async def start():
    global health_app_instance
    try:
        health_app_instance = HealthWellnessApp()
        await cl.Message(
            content="Assalam-o-Alaikum! Main aapka Health & Wellness Agent hoon. Aapki kya madad kar sakta hoon?",
        ).send()
        print("Chainlit: Chat session started and HealthWellnessApp initialized successfully.")
    except Exception as e:
        error_msg = f"Failed to initialize Health & Wellness App: {str(e)}. Please check your configuration (e.g., GEMINI_API_KEY)."
        print(f"Chainlit: Initialization Error: {error_msg}")
        traceback.print_exc()
        await cl.Message(
            content=error_msg,
            author="Error"
        ).send()
        health_app_instance = None

@cl.on_message
async def handle_message(message: cl.Message):
    global health_app_instance
    if health_app_instance is None:
        await cl.Message(
            content="Maaf kijiye, app abhi tayyar nahi hai. Baraye meherbani dobara shuru karein ya administrator se rabta karein.",
            author="Error"
        ).send()
        return

    print(f"Chainlit: User message received: {message.content}")

    msg = cl.Message(content="")
    await msg.send()

    try:
        response_text = await health_app_instance.process_user_input_async(
            message.content, 
            user_context={"chat_history": [msg.to_dict() for msg in cl.user_session.get("chat_history", [])]}
        )
        
        msg.content = response_text
        await msg.update()

        print(f"Chainlit: Model response sent: {response_text[:50]}...")

    except Exception as e:
        error_message = f"Maaf kijiye, jawab dene mein koi masla ho gaya hai: {e}. Baraye meherbani dobara koshish karein."
        print(f"Chainlit: Error during message processing: {e}")
        traceback.print_exc()
        
        msg.content = error_message
        await msg.update()
