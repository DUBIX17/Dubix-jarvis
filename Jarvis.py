from flask import Flask, request, jsonify
import requests
import os
import json

app = Flask(__name__)

MAX_HISTORY = 5
conversation_history = []
AI_BEHAVIOR_PROMPT = "You are a helpful assistant for home automation and coding projects."

# Replace with actual Gemini model
GEMINI_MODEL = "gemini-1.5-flash"

@app.route("/gemini_proxy", methods=["GET"])
def gemini_proxy():
    global conversation_history

    api_key = request.args.get("api_key")
    user_text = request.args.get("text")

    if not api_key or not user_text:
        return jsonify({"error": "Missing api_key or text"}), 400

    # Build payload in the correct role/parts format
    payload = []

    # First message: AI behavior prompt
    payload.append({
        "role": "user",
        "parts": [{"text": AI_BEHAVIOR_PROMPT}]
    })

    # Add last MAX_HISTORY conversation rounds
    for round_ in conversation_history[-MAX_HISTORY:]:
        user_msg, ai_msg = round_
        payload.append({
            "role": "user",
            "parts": [{"text": user_msg}]
        })
        payload.append({
            "role": "model",
            "parts": [{"text": ai_msg}]
        })

    # Current user message
    payload.append({
        "role": "user",
        "parts": [{"text": user_text}]
    })

    # POST request to Gemini with API key in URL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    data = {"conversation": payload}

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        gemini_data = response.json()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Extract AI reply
    ai_reply = ""
    if "conversation" in gemini_data and isinstance(gemini_data["conversation"], list):
        for msg in gemini_data["conversation"]:
            if msg.get("role") == "model" and "parts" in msg:
                for part in msg["parts"]:
                    ai_reply += part.get("text", "")

    # Save current round in history
    conversation_history.append((user_text, ai_reply))

    return jsonify({"reply": ai_reply})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
