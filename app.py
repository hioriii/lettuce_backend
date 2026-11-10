import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict_lettuce

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Ecomist Lettuce Detection API is running."
    })

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded."
        }), 400

    image = request.files["image"]

    if image.filename == "":
        return jsonify({
            "error": "Empty image filename."
        }), 400

    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    try:
        result = predict_lettuce(image_path)

        return jsonify({
            "status": "success",
            "prediction": result["result"],
            "confidence": result["confidence"]
        })

    except Exception as e:
        return jsonify({
            "status": "failed",
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)