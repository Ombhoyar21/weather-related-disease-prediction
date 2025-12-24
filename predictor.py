import joblib
import pandas as pd
import numpy as np
import os

class DiseasePredictor:
    def __init__(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.model = joblib.load(os.path.join(base_path, 'model (1).pkl'))
        self.scaler = joblib.load(os.path.join(base_path, 'scaler (2).pkl'))
        self.encoder = joblib.load(os.path.join(base_path, 'encoder (2).pkl'))
        
        # Features required by the scaler
        self.feature_names = [
            'Age', 'Gender', 'Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'nausea',
            'joint_pain', 'abdominal_pain', 'high_fever', 'chills', 'fatigue',
            'runny_nose', 'pain_behind_the_eyes', 'dizziness', 'headache', 'chest_pain',
            'vomiting', 'cough', 'shivering', 'asthma_history', 'high_cholesterol',
            'diabetes', 'obesity', 'hiv_aids', 'nasal_polyps', 'asthma',
            'high_blood_pressure', 'severe_headache', 'weakness', 'trouble_seeing',
            'fever', 'body_aches', 'sore_throat', 'sneezing', 'diarrhea',
            'rapid_breathing', 'rapid_heart_rate', 'swollen_glands', 'rashes',
            'sinus_headache', 'facial_pain', 'shortness_of_breath',
            'reduced_smell_and_taste', 'skin_irritation', 'itchiness',
            'throbbing_headache', 'confusion', 'back_pain', 'knee_ache'
        ]

    def predict(self, input_data):
        # input_data should be a dictionary with feature names as keys
        df = pd.DataFrame([input_data])
        
        # Ensure all features are present, fill missing with 0.0, and convert to numeric
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        # Reorder columns to match scaler
        df = df[self.feature_names]
        
        # Scale
        scaled_data = self.scaler.transform(df)
        
        # Predict (LinearRegression returns float)
        raw_prediction = self.model.predict(scaled_data)[0]
        
        # Determine class by rounding and clipping
        num_classes = len(self.encoder.classes_)
        class_idx = int(round(raw_prediction))
        class_idx = max(0, min(class_idx, num_classes - 1))
        
        # Calculate pseudo-confidence (inverse distance to nearest integer)
        dist = abs(raw_prediction - class_idx)
        # Scale dist [0, 0.5] to confidence [100, 50]
        confidence = max(50.0, (1.0 - (dist * 2)) * 100.0)
        
        # Inverse transform encoder
        disease = self.encoder.classes_[class_idx]
        
        return {
            "prognosis": str(disease),
            "confidence": round(confidence, 1)
        }

predictor = DiseasePredictor()
