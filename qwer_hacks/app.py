from flask import Flask, render_template, request, jsonify
from google import genai
from google.oauth2 import service_account
from google.cloud import vision
import os

app = Flask(__name__)

# creds = service_account.Credentials.from_service_account_file("vision-key.json")
# print(f"Service Account Email: {creds.service_account_email}")

# vision_client = vision.ImageAnnotatorClient(credentials=creds)  # ✅ THIS is what you should use
# client = genai.Client()
# print(f"Valid: {creds.valid}")

# Test creating client
# try:
#     vision_client = vision.ImageAnnotatorClient()
#     print("✅ Vision client created successfully")
# except Exception as e:
#     print(f"❌ Error: {e}")


vision_client = vision.ImageAnnotatorClient()
client = genai.Client()  # Uses GEMINI_API_KEY from environment

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    user_text = (data.get("message") or "").strip()
    if not user_text:
        return jsonify({"reply": "Please type a message."})

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=user_text
        )
        return jsonify({"reply": response.text})
    except Exception as e:
        return jsonify({"reply": f"Error calling Gemini: {e}"}), 500


def _vision_face_result(image_bytes: bytes):
    """Helper to call Vision and return (faces, error_message_or_none)."""
    try:
        image = vision.Image(content=image_bytes)
        response = vision_client.face_detection(image=image)

        print(response)

        if response.error and response.error.message:
            return None, response.error.message

        faces = response.face_annotations or []
        return faces, None
    except Exception as e:
        return None, str(e)


@app.route("/detect_emotion", methods=["POST"])
def detect_emotion():
    try:
        if "photo" not in request.files:
            return jsonify({"error": "No file uploaded. Use form field name 'photo'."}), 400

        file = request.files["photo"]
        image_bytes = file.read()

        faces, err = _vision_face_result(image_bytes)
        if err:
            return jsonify({"error": err}), 500
        if not faces:
            return jsonify({"faces": [], "message": "No face detected."}), 200

        f = faces[0]
        result = {
            "joy": f.joy_likelihood.name,
            "sorrow": f.sorrow_likelihood.name,
            "anger": f.anger_likelihood.name,
            "surprise": f.surprise_likelihood.name,
            "detection_confidence": f.detection_confidence,
        }
        return jsonify(result), 200

    except Exception as e:
        # Always JSON (prevents "Unexpected token <")
        return jsonify({"error": f"Server error in /detect_emotion: {e}"}), 500


@app.route("/detect_focus", methods=["POST"])
def detect_focus():
    try:
        if "photo" not in request.files:
            return jsonify({"error": "No file uploaded. Use form field name 'photo'."}), 400

        file = request.files["photo"]
        image_bytes = file.read()

        faces, err = _vision_face_result(image_bytes)
        if err:
            return jsonify({"error": err}), 500
        if not faces:
            return jsonify({"focused": False, "reason": "No face detected"}), 200

        f = faces[0]

        # Simple heuristic (replace later with something better)
        unfocused = (
            f.anger_likelihood.name in ("LIKELY", "VERY_LIKELY") or
            f.surprise_likelihood.name in ("LIKELY", "VERY_LIKELY")
        )

        return jsonify({
            "focused": not unfocused,
            "reason": "Face detected" if not unfocused else "High anger/surprise detected",
            "joy": f.joy_likelihood.name,
            "sorrow": f.sorrow_likelihood.name,
            "anger": f.anger_likelihood.name,
            "surprise": f.surprise_likelihood.name,
        }), 200

    except Exception as e:
        return jsonify({"error": f"Server error in /detect_focus: {e}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5400))
    # Turn on debug while developing so you can see the traceback in your console:
    app.run(host="0.0.0.0", port=port, debug=True)
