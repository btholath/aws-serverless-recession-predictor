"""
Inference Script for SageMaker XGBoost Endpoint
This file MUST be named 'inference.py' and included with the model artifact.
"""
import os
import json
import xgboost as xgb
import numpy as np


def model_fn(model_dir):
    """
    Load the XGBoost model from the model directory.
    SageMaker calls this function when the endpoint starts.
    
    Args:
        model_dir: Path to the directory containing model artifacts
        
    Returns:
        Loaded XGBoost Booster model
    """
    print(f"📦 Loading model from: {model_dir}")
    print(f"📁 Contents: {os.listdir(model_dir)}")
    
    model_path = os.path.join(model_dir, "xgboost-model")
    
    if not os.path.exists(model_path):
        # Try alternative names
        for name in ['model.xgb', 'model.bin', 'xgboost-model.bin']:
            alt_path = os.path.join(model_dir, name)
            if os.path.exists(alt_path):
                model_path = alt_path
                break
        else:
            raise FileNotFoundError(f"Model file not found in {model_dir}")
    
    model = xgb.Booster()
    model.load_model(model_path)
    print("✅ Model loaded successfully!")
    return model


def input_fn(request_body, request_content_type):
    """
    Parse the input request body.
    
    Args:
        request_body: The body of the request (CSV string)
        request_content_type: The content type of the request
        
    Returns:
        XGBoost DMatrix ready for prediction
    """
    print(f"📥 Received input (type: {request_content_type})")
    
    if request_content_type == "text/csv":
        # Parse CSV: "value1, value2, value3, value4"
        values = [float(x.strip()) for x in request_body.strip().split(",")]
        print(f"📊 Parsed values: {values}")
        return xgb.DMatrix(np.array([values]))
    
    elif request_content_type == "application/json":
        # Parse JSON: {"features": [value1, value2, value3, value4]}
        data = json.loads(request_body)
        features = data.get('features', data.get('data', data))
        if isinstance(features[0], list):
            # Batch prediction: [[v1, v2, ...], [v1, v2, ...]]
            return xgb.DMatrix(np.array(features))
        else:
            # Single prediction: [v1, v2, v3, v4]
            return xgb.DMatrix(np.array([features]))
    
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """
    Make predictions using the loaded model.
    
    Args:
        input_data: XGBoost DMatrix from input_fn
        model: Loaded XGBoost model from model_fn
        
    Returns:
        Numpy array of predictions
    """
    print("🔮 Making prediction...")
    prediction = model.predict(input_data)
    print(f"📈 Prediction: {prediction}")
    return prediction


def output_fn(prediction, accept):
    """
    Format the prediction output.
    
    Args:
        prediction: Numpy array from predict_fn
        accept: The accept header from the request
        
    Returns:
        Formatted response string
    """
    print(f"📤 Formatting output (accept: {accept})")
    
    if accept == "application/json":
        return json.dumps({"prediction": prediction.tolist()})
    else:
        # Default to CSV format
        if len(prediction) == 1:
            return str(prediction[0])
        return ",".join(map(str, prediction))
