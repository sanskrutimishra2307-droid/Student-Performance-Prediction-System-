"""
Test Suite for FastAPI REST Endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Student Performance Prediction API"

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_api_features():
    response = client.get("/api/v1/features")
    assert response.status_code == 200
    data = response.json()
    assert "selected_features" in data
    assert "classes" in data

def test_api_predict():
    payload = {
        "age": 17,
        "gender": "male",
        "school_type": "private",
        "parent_education": "diploma",
        "study_hours": 3.0,
        "attendance_percentage": 70.0,
        "internet_access": "yes",
        "travel_time": "15-30 min",
        "extra_activities": "no",
        "study_method": "textbook",
        "math_score": 55.0,
        "science_score": 58.0,
        "english_score": 60.0
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "prediction" in data
    assert "explanation" in data
