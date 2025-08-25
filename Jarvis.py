from flask import Flask, request, jsonify
import requests
import os
import json

app = Flask(__name__)

# Configurable: how many past conversation rounds to keep
MAX_HISTORY = 5

# In-memory conversation history
conversation_history = []

# AI behavior prompt (first message)
AI_BEHAVIOR_PROMPT = "You are a helpful assistant for home automation and coding projects."

# Google Gemini endpoint
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"  # replace with actual Gemini URL

@app.route("/gemini_proxy", methods=["GET"])
def gemini_proxy():
    global conversation_history

    # Get URL parameters
    api_key = request.args.get("api_key")
    user_text = request.args.get("text")

    if not api_key or not user_text:
        return jsonify({"error": "Missing api_key or text"}), 400

    # Build conversation payload
    payload = []

    # First message: AI behavior/system prompt
    payload.append({
        "role": "user",
        "parts": [{"text": AI_BEHAVIOR_PROMPT}]
    })

    # Include last MAX_HISTORY rounds
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

    # Build Gemini POST request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    post_data = {"conversation": payload}

    try:
        response = requests.post(GEMINI_URL, headers=headers, data=json.dumps(post_data))
        response.raise_for_status()
        gemini_data = response.json()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Extract AI reply from Gemini response
    ai_reply = ""
    if "output" in gemini_data:
        for item in gemini_data["output"]:
            if "parts" in item:
                for part in item["parts"]:
                    ai_reply += part.get("text", "")

    # Update conversation history
    conversation_history.append((user_text, ai_reply))

    return jsonify({"reply": ai_reply})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
