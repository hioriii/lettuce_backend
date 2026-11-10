import joblib

from utils.image_processing import process_image

MODEL_PATH = "model/lettuce_model.pkl"
LABELS_PATH = "model/labels.pkl"

model = joblib.load(MODEL_PATH)
labels = joblib.load(LABELS_PATH)

def predict_lettuce(image_path):
    image_data = process_image(image_path)
    image_data = [image_data]

    prediction = model.predict(image_data)[0]
    confidence = model.predict_proba(image_data)[0][prediction]

    return {
        "result": labels[prediction],
        "confidence": round(float(confidence) * 100, 2)
    }