"""
Flask backend for the Federated-Learning Skin Disease Prediction system.

Endpoints
---------
POST /api/auth/register        {name, email, password}
POST /api/auth/login           {email, password}
POST /api/predict              multipart image (+ optional user_id) -> prediction
GET  /api/history/<user_id>    prediction history for a user
POST /api/diet                 {height_cm, weight_kg} -> diet plan suggestion
GET  /api/classes              list of disease classes the model recognizes
GET  /api/health               simple health check
"""

import os
import io
import sys

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.db_models import db, User, Prediction, DietProfile
from models.cnn_model import SkinDiseaseCNN, CLASS_NAMES, CLASS_INFO
from utils.diet_plan import get_diet_plan

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "global_model.pth")

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)  # allow the static frontend (different origin/port) to call this API
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.join(BASE_DIR, 'skin_disease.db')}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload cap

db.init_app(app)

DEVICE = "cpu"
IMAGE_SIZE = 64

_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


def load_model():
    model = SkinDiseaseCNN().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded federated global model from {MODEL_PATH}")
    else:
        print(
            "WARNING: no trained model found at "
            f"{MODEL_PATH}. Run `python -m federated.simulate_training` first. "
            "Serving with a randomly-initialized model for now."
        )
    model.eval()
    return model


MODEL = load_model()


# ---------------------------------------------------------------- health --
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)})


@app.route("/api/classes", methods=["GET"])
def get_classes():
    return jsonify({
        "classes": [
            {"code": c, **CLASS_INFO[c]} for c in CLASS_NAMES
        ]
    })


# ------------------------------------------------------------------ auth --
@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json(force=True)
    name, email, password = data.get("name"), data.get("email"), data.get("password")
    if not all([name, email, password]):
        return jsonify({"error": "name, email and password are required"}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "An account with this email already exists"}), 409

    user = User(name=name, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "Registered successfully", "user": user.to_dict()}), 201


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    email, password = data.get("email"), data.get("password")
    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        return jsonify({"error": "Invalid email or password"}), 401
    return jsonify({"message": "Login successful", "user": user.to_dict()})


# -------------------------------------------------------------- predict --
@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded (field name must be 'image')"}), 400

    file = request.files["image"]
    user_id = request.form.get("user_id")

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not read image file"}), 400

    tensor = _transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = MODEL.predict_proba(tensor).squeeze(0).tolist()

    ranked = sorted(zip(CLASS_NAMES, probs), key=lambda x: x[1], reverse=True)
    top_class, top_conf = ranked[0]

    filename = f"{os.urandom(4).hex()}_{file.filename}"
    save_path = os.path.join(UPLOAD_DIR, filename)
    img.save(save_path)

    prediction = Prediction(
        user_id=int(user_id) if user_id else None,
        image_filename=filename,
        predicted_class=top_class,
        confidence=top_conf,
        all_probabilities={c: p for c, p in ranked},
    )
    db.session.add(prediction)
    db.session.commit()

    return jsonify({
        "prediction": {
            "class": top_class,
            "info": CLASS_INFO[top_class],
            "confidence": round(top_conf, 4),
        },
        "all_probabilities": [
            {"class": c, "info": CLASS_INFO[c], "probability": round(p, 4)} for c, p in ranked
        ],
        "record_id": prediction.id,
        "disclaimer": (
            "This is an academic-project prediction, not a medical diagnosis. "
            "Please consult a dermatologist for any real skin concern."
        ),
    })


@app.route("/api/history/<int:user_id>", methods=["GET"])
def history(user_id):
    records = (
        Prediction.query.filter_by(user_id=user_id)
        .order_by(Prediction.created_at.desc())
        .limit(50)
        .all()
    )
    return jsonify({"history": [r.to_dict() for r in records]})


# ----------------------------------------------------------------- diet --
@app.route("/api/diet", methods=["POST"])
def diet():
    data = request.get_json(force=True)
    try:
        height_cm = float(data.get("height_cm"))
        weight_kg = float(data.get("weight_kg"))
    except (TypeError, ValueError):
        return jsonify({"error": "height_cm and weight_kg must be numbers"}), 400

    if height_cm <= 0 or weight_kg <= 0:
        return jsonify({"error": "height_cm and weight_kg must be positive"}), 400

    plan = get_diet_plan(height_cm, weight_kg)

    user_id = data.get("user_id")
    profile = DietProfile(
        user_id=int(user_id) if user_id else None,
        height_cm=height_cm,
        weight_kg=weight_kg,
        bmi=plan["bmi"],
        category=plan["category"],
    )
    db.session.add(profile)
    db.session.commit()

    return jsonify(plan)


with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
