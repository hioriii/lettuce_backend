import os
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from utils.image_processing import process_image

DATASET_PATH = "dataset"
MODEL_PATH = "model/lettuce_model.pkl"
LABELS_PATH = "model/labels.pkl"

labels = ["healthy", "unhealthy"]

X = []
y = []

for label_index, label in enumerate(labels):
    folder_path = os.path.join(DATASET_PATH, label)

    for filename in os.listdir(folder_path):

        try:
            image_path = os.path.join(folder_path, filename)
            image_data = process_image(image_path)
            X.append(image_data)
            y.append(label_index)
        except Exception as e:
            print(f"Skipped {filename}: {e}")

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    raise Exception("No images found. Add images inside dataset/healthy and dataset/unhealthy.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

search = GridSearchCV(
    LogisticRegression(max_iter=1000, class_weight="balanced"),
    {"C": [0.01, 0.1, 1, 10, 100]},
    cv=5,
    scoring="accuracy",
)
search.fit(X_train, y_train)
model = search.best_estimator_

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

os.makedirs("model", exist_ok=True)

joblib.dump(model, MODEL_PATH)
joblib.dump(labels, LABELS_PATH)

print("Model trained successfully.")
print(f"Best C: {search.best_params_['C']}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion matrix (rows=actual, cols=predicted) [healthy, unhealthy]:")
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions, target_names=labels))
print(f"Saved model to {MODEL_PATH}")