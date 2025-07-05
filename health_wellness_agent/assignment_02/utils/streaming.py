import google.generativeai as genai
from typing import Dict, Any, Generator

class StreamingChat:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        genai.configure(api_key=config["api_key"])
        self.model = genai.GenerativeModel(config["model"])

    def _format_prompt(self, query: str, context: Dict = None) -> str:
        """
        Formats the prompt for the Gemini API.
        This is a placeholder; you might need to enhance this based on your agent's needs.
        """
        formatted_prompt = query
        if context:
            # Agar context mein koi relevant information hai, toh usko prompt mein shamil karein.
            # Example: formatted_prompt = f"Context: {context.get('user_goal', '')}\nQuery: {query}"
            pass # Abhi ke liye context ko prompt mein shamil nahi kar rahe, sirf placeholder hai.
        return formatted_prompt

    def stream_response(self, query: str, context: Dict = None) -> Generator[str, None, None]:
        """Stream response from Gemini API"""
        try:
            prompt = self._format_prompt(query, context)

            # Generate streaming response
            response = self.model.generate_content(
                prompt,
                stream=True
            )

            for chunk in response:
                if chunk.text:
                    yield chunk.text # .text lagana zaroori hai yahan
        except Exception as e:
            # Agar koi error aaye toh usko print karein aur empty string yield karein
            print(f"Error during streaming: {e}")
            yield f"Error: {e}" # User ko error message bhej dein