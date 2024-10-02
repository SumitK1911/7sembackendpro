import os
import json
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from PIL import Image
import speech_recognition as sr
import pyttsx3
import threading
import google.generativeai as genai  # Import Google Gemini
from typing import List
from pydantic import BaseModel


class AIVoiceAssistant:
    def __init__(self, vector_db_url, api_key):
        self.vector_db_client = QdrantClient(url=vector_db_url, prefer_grpc=False)
        self.collection_name = "image_vectors"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        self.engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        self.conversation_history = []
        self._welcome_message_spoken = threading.Event()
        self.history_file = "chat_history.json"
        self.is_speaking = threading.Event()  # Track if the assistant is currently speaking
        self.is_processing_query = threading.Event()  # Track if the assistant is processing a query
        self.load_chat_history()  # Load chat history on initialization
    
    def extract_features(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            outputs = self.model.get_image_features(pixel_values=inputs['pixel_values'])
            features = outputs.detach().numpy().flatten()
            return features
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None

    def query_vector_db(self, query_text, top_k=3):
        try:
            inputs = self.processor(text=query_text, return_tensors="pt")
            text_features = self.model.get_text_features(input_ids=inputs['input_ids'])
            text_features = text_features.detach().numpy().flatten()
            
            search_result = self.vector_db_client.search(
                collection_name=self.collection_name,
                query_vector=text_features,
                limit=top_k
            )
            return search_result
        except Exception as e:
            print(f"Error querying vector database: {e}")
            return []

    def generate_response(self, query_text, action_result=None,additional_data=None):
        search_results = self.query_vector_db(query_text)

        if not search_results:
            response_text = "It seems like your query is out of context. Could you please clarify or ask something else?"
            self.conversation_history.append({"role": "model", "parts": response_text})
            return response_text

        combined_metadata = " ".join([result.payload.get("description", "") for result in search_results])

        if action_result == "added_to_cart":
            prompt = f"You've added the following item to your cart: {combined_metadata}. Would you like to continue shopping or proceed to checkout?"
        elif action_result == "deleted_from_cart":
            prompt = f"The item '{search_results[0].payload.get('description')}' has been removed from your cart. Is there anything else you'd like to do?"
        elif action_result == "discount_applied":
        # Use the data from `additional_data` to generate the discount message
          if additional_data:
            discount = additional_data.get('discount')
            final_amount = additional_data.get('final_amount')
            prompt = f"A discount of {discount}% has been applied. The total is now {final_amount:.2f}."
          else:
            prompt = "A discount has been applied." 
        else:
            prompt = f"Based on your request, I found the following items: {combined_metadata}. How can I assist you further?"

        interaction_guidelines = """
        You are an AI assistant of a retail shop, a shop known for its wonderful product. Your role is to interact with customers and help them with their inquiries.

        Guidelines for interaction:
        - Start the conversation by greeting the customer and asking how you can assist them.
        - If a customer asks about you, provide a brief introduction about cloth store
        - If a customer asks about the item or product, provide a brief overview of popular items or products.
        - If a customer gives their order, confirm it by repeating back to them.
        - Provide concise, relevant responses that relate directly to the customer's query.
        - If you don't have an answer, apologize and suggest that they speak to a human assistant for further help.
        - Always be polite and end the conversation with a friendly farewell.
        - Do not generate additional dialogue beyond what is necessary to address the customer's current inquiry. Focus on responding directly to what the customer has said.

        Don't reply randomly, just reply based on the data within the database and collection.
        """

        full_prompt = f"{interaction_guidelines}\n\n{prompt}"

        self.conversation_history.append({"role": "user", "parts": query_text})
        self.conversation_history.append({"role": "model", "parts": full_prompt})

        try:
            chat = self.gemini_model.start_chat(
                history=[{"role": item["role"], "parts": item["parts"]} for item in self.conversation_history]
            )
            response = chat.send_message(full_prompt)
            response_text = response.text.strip().replace("*", "")
            self.conversation_history.append({"role": "model", "parts": response_text})

            if "how can I assist you today" in response_text.lower():
                return "Is there anything specific you need help with today?"
            
            return response_text
    
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, I encountered an error while trying to respond to your request."

    def handle_user_query(self, query_text):
        self.stop_listening()
        response_text = self.generate_response(query_text)
        self.speak_text(response_text)  
        self.start_listening()
        return response_text
    
    def speak_text(self, text):
        if text:
            self.is_speaking.set()  
            self.engine.say(text)
            self.engine.runAndWait()
            self.is_speaking.clear()  

    def speak_welcome_message(self):
        if not self._welcome_message_spoken.is_set():
            welcome_text = "Welcome to AIVoice Assistance. Feel free to shop cloth in our RetailShop. How can I assist you today?"
            self.speak_text(welcome_text)
            self._welcome_message_spoken.set()
    
    def listen_to_user(self):
        try:
            with sr.Microphone() as source:
                print("Listening...")
                audio = self.recognizer.listen(source)
                query_text = self.recognizer.recognize_google(audio)
                print(f"User: {query_text}")
                return query_text
        except sr.UnknownValueError:
            return "Sorry, I didn't catch that."
        except sr.RequestError:
            return "Sorry, there was an issue with the speech recognition service."

    def start_listening(self):
        if not self.is_speaking.is_set() and not self.is_processing_query.is_set():
            threading.Thread(target=self.listen_to_user).start()  

    def stop_listening(self):
        if self.recognizer:
           
            pass

    def save_chat_history(self):
        try:
            with open(self.history_file, 'w') as file:
                json.dump(self.conversation_history, file, indent=4)
        except Exception as e:
            print(f"Error saving chat history: {e}")

    def load_chat_history(self):
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as file:
                    self.conversation_history = json.load(file)
            else:
                self.conversation_history = []
        except Exception as e:
            print(f"Error loading chat history: {e}")

if __name__ == "__main__":
    vector_db_url = "http://localhost:6333"
    api_key = os.getenv("API_KEY")
    assistant = AIVoiceAssistant(vector_db_url, api_key)
    assistant.speak_welcome_message()
    
    while True:
        user_query = assistant.listen_to_user()
        if "terminate" in user_query.lower():
            print("Terminating...")
            assistant.speak_text("Goodbye!")
            break
        
        response = assistant.handle_user_query(user_query)
        print(f"AI Response: {response}")
