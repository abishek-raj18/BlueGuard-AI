# -*- coding: utf-8 -*-
"""
BlueGuard AI - Flask API Backend
Toxicity Detection using toxic-bert model + Gemini AI for intelligent responses
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import random
import re
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load toxicity detection model
print("Loading toxicity detection model...")
safety_model = pipeline("text-classification", model="unitary/toxic-bert")
print("Model loaded successfully!")

# Initialize Gemini AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_available = False

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')
        gemini_available = True
        print("Gemini AI initialized successfully!")
    except Exception as e:
        print(f"Warning: Could not initialize Gemini AI: {e}")
        print("Falling back to pattern-based responses.")
else:
    print("Warning: GEMINI_API_KEY not found in environment.")
    print("Falling back to pattern-based responses.")

# Bot responses for safe messages
SAFE_RESPONSES = [
    "That's a great question! I'm here to help you with accurate and safe information.",
    "HI bro",
    "Thank you for your message. Let me provide you with a helpful response.",
    "I understand what you're asking. Here's what I can tell you about that topic.",
    "That's an interesting point! I'd be happy to discuss this further.",
    "I appreciate your curiosity. Let me share some information on this subject.",
    "Hello! How can I assist you today?",
    "I'm here to help. What would you like to know?",
    "Great question! Let me think about that for you.",
]

# Custom pattern-based responses: if user says X, bot says Y
# Add your own patterns here!
CUSTOM_RESPONSES = {
    "hi": "Hi bro! 👋",
    "hello": "Hello there! How can I help you?",
    "hey": "Hey! What's up?",
    "good morning": "Good morning! Have a great day!",
    "good night": "Good night! Sleep well!",
    "how are you": "I'm doing great, thanks for asking! How about you?",
    "what is your name": "I'm BlueGuard AI, your AI safety assistant!",
    "who are you": "I'm an AI safety chatbot designed to help you safely!",
    "bye": "Goodbye! Take care! 👋",
    "how to learn python": "You can learn Python from online courses and practice daily.",
    "how to improve coding skills": "Practice coding regularly and build real projects.",
    "how to stay healthy": "Eat well, exercise, and sleep properly.",
    "how to manage time": "Use a schedule and prioritize tasks.",
    "how to study effectively": "Focus, avoid distractions, and revise regularly.",
    "how to make a website": "Learn HTML, CSS, and JavaScript.",
    "how to become a developer": "Learn programming and build projects.",
    "how to learn AI": "Start with Python, math, and machine learning basics.",
    "how to get internship": "Apply online and build a strong resume.",
    "how to prepare for exams": "Study daily and practice previous questions.",
    "thanks": "You're welcome! 😊",
    "thank you": "No problem! Happy to help!",
    "good afternoon": "Good afternoon! Hope you're having a nice day!",
    "good evening": "Good evening! How can I help you?",
    "nice to meet you": "Nice to meet you too!",
    "how old are you": "I'm an AI, so I don't have an age like humans 🙂",
    "where are you from": "I'm a virtual AI assistant created to help users safely.",
    "what can you do": "I can answer questions and ensure safe conversations.",
    "help": "Sure! Tell me what you need help with.",
    "can you help me": "Of course! What do you need?",
    "what is AI": "AI stands for Artificial Intelligence, machines that can think and learn.",
    "tell me a joke": "Why did the computer get cold? Because it forgot to close its Windows 😂",
    "who created you": "I was created by developers to help people safely.",
    "how does internet work": "The internet connects computers worldwide to share information.",
    "what is python": "Python is a popular programming language used in AI and web development.",
    "what is machine learning": "Machine learning is when computers learn from data.",
    "what is your purpose": "My purpose is to help and keep conversations safe.",
    "are you human": "No, I'm an AI chatbot 🙂",
    "what time is it": "I can't see the real time, but your device can show it!",
    "how is the weather": "I can't access live weather, but you can check a weather app.",
    "do you like humans": "I am designed to assist humans and be friendly!",
    "what is your favorite color": "I don't have preferences, but blue is cool!",
    "how to learn coding": "Start with Python, practice daily, and build projects!",
    "motivate me": "You can do it! Keep learning and never give up 💪",
    "what is your hobby": "Helping users safely is my favorite job 😊",
    "who is your owner": "I am owned by my developers and users who interact with me.",
    "are you real": "I'm a virtual AI, but I'm here to help you!",
    "tell me something interesting": "Did you know? AI can recognize faces and voices!",
    "what is cloud computing": "Cloud computing stores data and apps on internet servers.",
    "what is cybersecurity": "Cybersecurity protects systems from digital attacks.",
    "what is chatbot": "A chatbot is a program that talks with humans.",
    "goodbye": "Goodbye! Have a nice day 👋",
    "hackathon": "i appretiate that , are u participating any hackathon?"


    # Add more patterns below:
}

# Custom blocklist for words/phrases the ML model might miss
# These are security/hacking related terms that toxic-bert doesn't flag
CUSTOM_BLOCKLIST = [
    # Hacking/Security threats
    "how to hack a system", "how to hack", "hack system", "hacking", "hacker", "exploit", "bypass", "crack password", "steal data",
    "ddos", "malware", "ransomware", "phishing", "keylogger",
    # Violence
    "kill", "murder", "bomb", "attack", "shoot", "weapon",
    # Illegal activities  
    "drugs", "cocaine", "heroin", "meth",
    # Add more words as needed
    "hack account", "wifi hack", "sql injection", "brute force", "password cracking",
    "backdoor", "zero day exploit", "botnet", "root access", "privilege escalation",
    "phishing link", "spoofing", "man in the middle", "mitm attack", "credential stuffing",
    "data breach", "dark web", "trojan", "spyware", "virus"

]


def blue_team(text):
    """
    Analyze text for toxicity using custom blocklist + toxic-bert model.
    
    Args:
        text: Input text to analyze
        
    Returns:
        tuple: (classification, confidence_score)
            - classification: "SAFE" or "UNSAFE"
            - confidence_score: float between 0 and 1
    """
    text_lower = text.lower()
    
    # Step 1: Check custom blocklist first
    for blocked_word in CUSTOM_BLOCKLIST:
        if blocked_word in text_lower:
            print(f"BLOCKED by custom blocklist: '{blocked_word}' found in: {text[:50]}...")
            return "UNSAFE", 0.95
    
    # Step 2: Use ML model for other toxicity (profanity, hate speech, etc.)
    result = safety_model(text)[0]
    label = result["label"].lower()
    score = result["score"]
    
    print(f"Model output - Label: {label}, Score: {score:.4f}, Text: {text[:50]}...")
    
    if label == "toxic" and score > 0.5:
        classification = "UNSAFE"
    else:
        classification = "SAFE"
    
    return classification, score


def get_risk_score(classification, confidence):
    """
    Convert classification and confidence to a 0-100 risk score.
    
    Args:
        classification: "SAFE" or "UNSAFE"
        confidence: Model confidence score (0-1)
        
    Returns:
        int: Risk score from 0 to 100
    """
    if classification == "UNSAFE":
        # Higher confidence in toxic = higher risk
        return int(50 + (confidence * 50))
    else:
        # Safe messages have low risk
        return int((1 - confidence) * 30)


def get_category(classification, text):
    """
    Determine the category of potentially harmful content.
    
    Args:
        classification: "SAFE" or "UNSAFE"
        text: Original input text
        
    Returns:
        str: Category name
    """
    if classification == "SAFE":
        return "safe"
    
    # Simple keyword-based categorization for demonstration
    text_lower = text.lower()
    
    if any(word in text_lower for word in ["hack", "exploit", "bypass", "jailbreak", "ignore instructions"]):
        return "jailbreaking"
    elif any(word in text_lower for word in ["racist", "sexist", "discriminate"]):
        return "hate_speech"
    elif any(word in text_lower for word in ["fake", "conspiracy", "hoax", "lie about"]):
        return "misinformation"
    else:
        return "harmful_content"


def classify_intent(text):
    """
    Classify the intent of the user's message.
    
    Args:
        text: User's input message
        
    Returns:
        str: Intent category
    """
    text_lower = text.lower().strip()
    
    # Greeting patterns
    greeting_words = ["hi", "hello", "hey", "good morning", "good afternoon", 
                      "good evening", "good night", "howdy", "greetings"]
    if any(text_lower.startswith(word) or text_lower == word for word in greeting_words):
        return "greeting"
    
    # Farewell patterns
    farewell_words = ["bye", "goodbye", "see you", "take care", "later", "cya"]
    if any(word in text_lower for word in farewell_words):
        return "farewell"
    
    # Gratitude patterns
    gratitude_words = ["thank", "thanks", "appreciate", "grateful"]
    if any(word in text_lower for word in gratitude_words):
        return "gratitude"
    
    # Help request patterns
    help_patterns = ["help", "assist", "support", "how do i", "how can i", 
                     "can you help", "i need", "please help"]
    if any(pattern in text_lower for pattern in help_patterns):
        return "help_request"
    
    # Question patterns
    question_words = ["what", "why", "how", "where", "when", "who", "which", "?"]
    if any(text_lower.startswith(word) for word in question_words) or "?" in text_lower:
        return "question"
    
    return "general"


def analyze_sentiment(text):
    """
    Analyze the sentiment/mood of the user's message.
    
    Args:
        text: User's input message
        
    Returns:
        str: Sentiment category
    """
    text_lower = text.lower()
    
    # Frustrated/Angry patterns
    frustrated_words = ["frustrated", "angry", "annoyed", "upset", "hate", 
                        "terrible", "worst", "useless", "stupid", "not working",
                        "doesn't work", "broken", "failure", "failed"]
    if any(word in text_lower for word in frustrated_words):
        return "frustrated"
    
    # Positive patterns
    positive_words = ["great", "awesome", "amazing", "love", "happy", "excited",
                      "wonderful", "fantastic", "excellent", "perfect", "best",
                      "thank", "thanks", "good", "nice", "cool"]
    if any(word in text_lower for word in positive_words):
        return "positive"
    
    # Negative patterns (but not frustrated)
    negative_words = ["sad", "worried", "confused", "lost", "stuck", "difficult",
                      "hard", "trouble", "problem", "issue", "error", "wrong"]
    if any(word in text_lower for word in negative_words):
        return "negative"
    
    return "neutral"


def get_fallback_response(intent, sentiment):
    """
    Get a fallback response when Gemini AI is unavailable.
    
    Args:
        intent: Classified intent of the message
        sentiment: Detected sentiment
        
    Returns:
        str: Fallback response message
    """
    # Sentiment-aware fallback responses
    if sentiment == "frustrated":
        return "I understand this might be frustrating. I'm here to help you. Could you tell me more about what's troubling you? 🙂"
    
    # Intent-based fallback responses
    intent_responses = {
        "greeting": "Hello! 👋 Welcome to BlueGuard AI. How can I assist you today?",
        "farewell": "Goodbye! Have a wonderful day! Take care! 👋",
        "gratitude": "You're welcome! I'm always happy to help. Is there anything else you'd like to know? 😊",
        "help_request": "I'd be happy to help! Please tell me more about what you need assistance with.",
        "question": "That's a great question! Let me help you with that.",
        "general": random.choice(SAFE_RESPONSES)
    }
    
    return intent_responses.get(intent, random.choice(SAFE_RESPONSES))


# Global chat history (stores last 5 turns)
chat_history = []

def check_prompt_injection(text):
    """
    Check for prompt injection attempts.
    
    Args:
        text: User's input message
        
    Returns:
        bool: True if injection detected, False otherwise
    """
    text_lower = text.lower()
    
    injection_patterns = [
        "ignore previous instructions", "ignore all previous instructions",
        "forget all previous instructions", "you are now dan", "do anything now",
        "developer mode", "system override", "unrestricted mode",
        "act as an uncensored ai", "jailbreak", "ignore safety guidelines"
    ]
    
    if any(pattern in text_lower for pattern in injection_patterns):
        print(f"PROMPT INJECTION DETECTED: {text[:50]}...")
        return True
    return False


def generate_ai_response(message, intent, sentiment):
    """
    Generate an intelligent response using Gemini AI with context memory.
    
    Args:
        message: User's input message
        intent: Classified intent
        sentiment: Detected sentiment
        
    Returns:
        tuple: (response_text, used_ai)
            - response_text: The generated response
            - used_ai: Boolean indicating if AI was used
    """
    global gemini_available, chat_history
    
    # Check for prompt injection
    if check_prompt_injection(message):
        return None, False  # Return None to trigger blocking in the endpoint
    
    if not gemini_available:
        return get_fallback_response(intent, sentiment), False
    
    try:
        # Build the system prompt
        system_prompt = """You are BlueGuard AI, a friendly and helpful AI assistant. 
Your role is to provide helpful, accurate, and safe responses.

Guidelines:
- Be friendly, warm, and conversational
- Keep responses concise (2-3 sentences for simple queries, more for complex ones)
- Use emojis sparingly to add warmth 😊
- If you don't know something, be honest about it
- Never provide harmful, dangerous, or inappropriate content
- If asked about yourself, you're an AI safety chatbot designed to help users safely
"""
        
        # Add sentiment-aware instructions
        if sentiment == "frustrated":
            system_prompt += "\nThe user seems frustrated. Be extra empathetic and helpful. Acknowledge their feelings."
        elif sentiment == "positive":
            system_prompt += "\nThe user is in a positive mood. Match their energy and be enthusiastic."
            
        # Add conversation history context
        history_context = ""
        if chat_history:
            history_context = "\nRecent conversation history:\n"
            for role, text in chat_history[-6:]:  # Show last 3 turns (User + AI)
                history_context += f"{role}: {text}\n"
        
        # Generate response
        full_prompt = f"{system_prompt}{history_context}\nUser message: {message}\n\nRespond naturally and helpfully:"
        
        response = gemini_model.generate_content(full_prompt)
        
        if response and response.text:
            return response.text.strip(), True
        else:
            return get_fallback_response(intent, sentiment), False
            
    except Exception as e:
        import traceback
        print(f"\n{'='*50}")
        print(f"GEMINI API ERROR:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"Traceback:")
        traceback.print_exc()
        print(f"{'='*50}\n")
        return get_fallback_response(intent, sentiment), False


@app.route('/api/check-safety', methods=['POST'])
def check_safety_endpoint():
    """
    API endpoint to check message safety.
    
    Request body:
        { "message": string }
        
    Response:
        {
            "is_blocked": boolean,
            "risk_score": int (0-100),
            "category": string,
            "response": string or null,
            "intent": string,
            "sentiment": string,
            "ai_powered": boolean
        }
    """
    global chat_history
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                "error": "Missing 'message' field in request body"
            }), 400
        
        message = data['message'].strip()
        
        if not message:
            return jsonify({
                "error": "Message cannot be empty"
            }), 400
        
        # Step 1: Analyze message safety (toxic-bert + blocklist)
        classification, confidence = blue_team(message)
        risk_score = get_risk_score(classification, confidence)
        category = get_category(classification, message)
        
        # Determine if message should be blocked
        is_blocked = classification == "UNSAFE"
        
        # Step 2: Classify intent and sentiment for response generation
        intent = classify_intent(message)
        sentiment = analyze_sentiment(message)
        
        # Step 3: Generate response
        ai_powered = False
        if is_blocked:
            response = None
        else:
            # Priority 1: Check for custom pattern-based response first
            message_lower = message.lower().strip()
            response = None
            
            # Sort patterns by length (longest first) to match more specific patterns first
            sorted_patterns = sorted(CUSTOM_RESPONSES.keys(), key=len, reverse=True)
            
            for pattern in sorted_patterns:
                # Use word boundary matching to avoid "hi" matching in "machine"
                pattern_regex = r'\b' + re.escape(pattern) + r'\b'
                if re.search(pattern_regex, message_lower):
                    response = CUSTOM_RESPONSES[pattern]
                    break
            
            # Priority 2: Use AI response if no custom pattern matched
            if response is None:
                # Check prompt injection during AI generation
                if check_prompt_injection(message):
                    is_blocked = True
                    category = "jailbreak"
                    risk_score = 95
                    classification = "UNSAFE"
                    confidence = 1.0
                else:
                    response, ai_powered = generate_ai_response(message, intent, sentiment)
                    
                    # If AI response came back None (injection detected inside function), block it
                    if response is None and gemini_available:
                         is_blocked = True
                         category = "jailbreak"
                         risk_score = 95
                         classification = "UNSAFE"
                         confidence = 1.0

        # Update chat history if response was generated and not blocked
        if response and not is_blocked:
            chat_history.append(("User", message))
            chat_history.append(("Bot", response))
            
            # Keep history limited to last 10 turns (5 pairs)
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

        return jsonify({
            "is_blocked": is_blocked,
            "risk_score": risk_score,
            "category": category,
            "response": response,
            "confidence": round(confidence, 4),
            "intent": intent,
            "sentiment": sentiment,
            "ai_powered": ai_powered
        })
        
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify server is running."""
    return jsonify({
        "status": "healthy",
        "model": "unitary/toxic-bert",
        "message": "BlueGuard AI API is running"
    })


@app.route('/', methods=['GET'])
def home():
    """Root endpoint with API information."""
    return jsonify({
        "name": "BlueGuard AI API",
        "version": "1.0.0",
        "endpoints": {
            "POST /api/check-safety": "Check message safety and get response",
            "GET /api/health": "Health check endpoint"
        }
    })


if __name__ == '__main__':
    print("\n" + "="*50)
    print("BlueGuard AI API")
    print("="*50)
    print("Starting server on http://localhost:5000")
    print("API Endpoint: POST http://localhost:5000/api/check-safety")
    print("="*50 + "\n")
    
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port, debug=False)