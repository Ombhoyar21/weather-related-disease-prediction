import joblib
import pandas as pd

try:
    model = joblib.load('model (1).pkl')
    scaler = joblib.load('scaler (2).pkl')
    encoder = joblib.load('encoder (2).pkl')
    
    print("Model type:", type(model))
    print("Scaler type:", type(scaler))
    print("Encoder type:", type(encoder))
    
    if hasattr(model, 'feature_names_in_'):
        print("Model Feature Names:", model.feature_names_in_)
    else:
        print("Model does not have feature_names_in_")
        
    if hasattr(scaler, 'feature_names_in_'):
        print("Scaler Feature Names:", scaler.feature_names_in_)

    if hasattr(encoder, 'classes_'):
        print("Encoder classes:", encoder.classes_)

except Exception as e:
    print("Error:", e)
