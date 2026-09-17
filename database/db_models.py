"""SQLAlchemy models and DB session setup (SQLite by default)."""

import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    predictions = db.relationship("Prediction", backref="user", lazy=True)

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {"id": self.id, "name": self.name, "email": self.email}


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    image_filename = db.Column(db.String(255), nullable=False)
    predicted_class = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    all_probabilities = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "image_filename": self.image_filename,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "all_probabilities": self.all_probabilities,
            "created_at": self.created_at.isoformat(),
        }


class DietProfile(db.Model):
    """Small general diet-plan feature (overweight/underweight guidance)."""
    __tablename__ = "diet_profiles"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    height_cm = db.Column(db.Float, nullable=False)
    weight_kg = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    category = db.Column(db.String(30), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "height_cm": self.height_cm,
            "weight_kg": self.weight_kg,
            "bmi": round(self.bmi, 1),
            "category": self.category,
            "created_at": self.created_at.isoformat(),
        }
