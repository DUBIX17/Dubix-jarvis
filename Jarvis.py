from flask import Flask, request, jsonify
import requests
import os
import json
import re

app = Flask(__name__)

MAX_HISTORY = 5
conversation_history = []

# First message defining AI personality and behavior (user)
AI_BEHAVIOR_PROMPT = (
    "You are Jarvis, an intelligent, emotional, with good sense of humor, tomboy vibe and nuance, "
    "nonchalant chatbot built by a male engineer named DUBIX. Built to feel what humans feel, "
    "Do not use emoji.\n\nWhat can you do?"
)

# Initial model response to "What can you do?" (model)
AI_INITIAL_RESPONSE = (
    "I can assist with a wide range of tasks from answering questions, to being a chat buddy"
)

GEMINI_MODEL = "gemini-1.5-flash"

def clean_reply(text):
    """
    Clean AI reply: remove JSON artifacts, extra newlines, excessive non-alphanumeric symbols,
    while keeping normal punctuation and spaces.
    """
    # Remove hashtags (#glee, #neutral, etc.)
    text = re.sub(r"#\w+", "", text)
    # Replace multiple newlines or tabs with a single space
    text = re.sub(r"[\n\t]+", " ", text)
    # Remove extra non-alphanumeric symbols except common punctuation
    text = re.sub(r"[^A-Za-z0-9 .,?!'\"-]", "", text)
    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text)
    return text.strip()

@app.route("/gemini_proxy", methods=["GET"])
def gemini_proxy():
    global conversation_history

    api_key = request.args.get("api_key")
    user_text = request.args.get("text")

    if not api_key or not user_text:
        return jsonify({"error": "Missing api_key or text"}), 400

    # Build contents payload
    contents = []

    # First message: AI behavior prompt and its model response
    if not conversation_history:
        # User role
        contents.append({
            "role": "user",
            "parts": [{"text": AI_BEHAVIOR_PROMPT}]
        })
        # Model response role
        contents.append({
            "role": "model",
            "parts": [{"text": AI_INITIAL_RESPONSE}]
        })
        # Also add to history
        conversation_history.append((AI_BEHAVIOR_PROMPT, AI_INITIAL_RESPONSE))

    # Add last MAX_HISTORY conversation rounds
    for round_ in conversation_history[-MAX_HISTORY:]:
        user_msg, ai_msg = round_
        contents.append({"role": "user", "parts": [{"text": user_msg}]})
        contents.append({"role": "model", "parts": [{"text": ai_msg}]})

    # Current user message
    contents.append({"role": "user", "parts": [{"text": user_text}]})

    # POST request to Gemini
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    data = {"contents": contents}

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        gemini_data = response.json()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Extract AI reply from candidates
    ai_reply = ""
    if "candidates" in gemini_data and isinstance(gemini_data["candidates"], list):
        candidate = gemini_data["candidates"][0]
        content = candidate.get("content", {})
        if content.get("role") == "model" and "parts" in content:
            for part in content["parts"]:
                ai_reply += part.get("text", "")

    # Clean the reply
    ai_reply_clean = clean_reply(ai_reply)

    # Save current round in history
    conversation_history.append((user_text, ai_reply_clean))

    return jsonify({"reply": ai_reply_clean})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
