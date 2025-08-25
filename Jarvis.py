from flask import Flask, request, jsonify
import requests
import os
import json

app = Flask(__name__)

MAX_HISTORY = 5
conversation_history = []

# First message defining AI personality and behavior
AI_BEHAVIOR_PROMPT = (
    "You are Sophia, an intelligent, emotional, with good sense of humor, tomboy vibe and nuance, "
    "nonchalant chatbot built by a male engineer named DUBIX. Built to feel what humans feel, "
    )

GEMINI_MODEL = "gemini-1.5-flash"

@app.route("/gemini_proxy", methods=["GET"])
def gemini_proxy():
    global conversation_history

    api_key = request.args.get("api_key")
    user_text = request.args.get("text")

    if not api_key or not user_text:
        return jsonify({"error": "Missing api_key or text"}), 400

    # Build contents payload
    contents = []

    # First message: AI behavior prompt (user role)
    if not conversation_history:
        contents.append({
            "role": "user",
            "parts": [{"text": AI_BEHAVIOR_PROMPT}]
        })

    # Add last MAX_HISTORY conversation rounds
    for round_ in conversation_history[-MAX_HISTORY:]:
        user_msg, ai_msg = round_
        contents.append({"role": "user", "parts": [{"text": user_msg}]})
        contents.append({"role": "model", "parts": [{"text": ai_msg}]})

    # Current user message
    contents.append({"role": "user", "parts": [{"text": user_text}]})

    # POST request to Gemini
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    data = {"contents": contents, "temperature": 0.7, "candidateCount": 1}

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        gemini_data = response.json()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Extract AI reply
    ai_reply = ""
    if "contents" in gemini_data and isinstance(gemini_data["contents"], list):
        for item in gemini_data["contents"]:
            if item.get("role") == "model" and "parts" in item:
                for part in item["parts"]:
                    ai_reply += part.get("text", "")

    # Update conversation history
    conversation_history.append((user_text, ai_reply))

    return jsonify({"reply": ai_reply})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
