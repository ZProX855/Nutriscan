import os
import json
import re

import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Load environment variables from .env (local dev) or system env (production)
load_dotenv()

app = Flask(__name__)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Vision-capable models tried in order of preference.
# DeepSeek V3 is text-only and cannot process images.
VISION_MODELS = [
    "google/gemini-2.0-flash-001",
    "google/gemini-flash-1.5",
    "meta-llama/llama-4-maverick",
]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if not OPENROUTER_API_KEY:
        return jsonify({"error": "OPENROUTER_API_KEY is not set. See README for setup instructions."}), 500

    try:
        data = request.get_json()
        image_data = data.get("image")
        mass = data.get("mass", 100)
        unit = data.get("unit", "g")

        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        # Convert mass to grams
        mass_g = float(mass) * 1000 if unit == "kg" else float(mass)

        # Strip base64 prefix and detect MIME type
        mime_type = "image/jpeg"
        raw_b64 = image_data
        if "," in image_data:
            header, raw_b64 = image_data.split(",", 1)
            if "png" in header:
                mime_type = "image/png"
            elif "webp" in header:
                mime_type = "image/webp"

        prompt = f"""You are an expert nutritionist and food scientist. Analyze this food image and provide highly accurate nutritional information.

The food portion weighs exactly {mass_g}g ({mass}{unit}).

Identify the food(s) visible and calculate precise macros and nutrition facts for this exact weight.

Respond ONLY with a valid JSON object in this exact format (no markdown, no explanation, no backticks):
{{
  "food_name": "specific food name",
  "food_description": "brief description of what you see",
  "confidence": 95,
  "serving_weight_g": {mass_g},
  "calories": 0,
  "macros": {{
    "protein_g": 0.0,
    "carbohydrates_g": 0.0,
    "fat_g": 0.0,
    "fiber_g": 0.0,
    "sugar_g": 0.0,
    "saturated_fat_g": 0.0,
    "unsaturated_fat_g": 0.0
  }},
  "micronutrients": {{
    "sodium_mg": 0.0,
    "potassium_mg": 0.0,
    "calcium_mg": 0.0,
    "iron_mg": 0.0,
    "vitamin_c_mg": 0.0,
    "vitamin_a_iu": 0.0
  }},
  "glycemic_index": 0,
  "water_content_g": 0.0,
  "health_score": 0,
  "tags": ["tag1", "tag2"]
}}

Be as accurate as possible using standard nutritional databases. All values must be for the specified weight of {mass_g}g."""

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        last_error = None
        for model in VISION_MODELS:
            try:
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mime_type};base64,{raw_b64}"},
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                    "max_tokens": 1000,
                }

                api_response = requests.post(
                    OPENROUTER_URL, headers=headers, json=payload, timeout=60
                )
                api_response.raise_for_status()

                result = api_response.json()
                response_text = result["choices"][0]["message"]["content"].strip()

                # Strip any accidental markdown fences
                response_text = re.sub(r"```json\n?", "", response_text)
                response_text = re.sub(r"```\n?", "", response_text).strip()

                nutrition_data = json.loads(response_text)
                return jsonify({"success": True, "data": nutrition_data})

            except Exception as e:
                last_error = e
                continue

        return jsonify({"error": f"All vision models failed. Last error: {str(last_error)}"}), 500

    except json.JSONDecodeError as e:
        return jsonify({"error": f"Failed to parse nutrition data: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=debug, host="0.0.0.0", port=port)
