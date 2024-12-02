from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
# Add this to allow all origins during development
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Rest of your existing code remains the same
# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model
MODEL_PATH = 'best_model.keras'
model = load_model(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(224, 224)):
    try:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Debug print
        print(f"Image preprocessed. Shape: {img_array.shape}")
        
        return img_array
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        raise
@app.route('/api/detect', methods=['POST'])
def detect_eye_flu():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file received'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
           # Debug prediction
        print(f"Prediction raw output: {prediction}")
        print(f"Prediction processed: {bool(prediction[0][0] > 0.5)}")
        # Save the uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Preprocess the image
            preprocessed_image = preprocess_image(filepath)
            
            # Make prediction
            prediction = model.predict(preprocessed_image)
            
            # Clean up - remove the temporary file
            os.remove(filepath)
            
            # Return result
            return jsonify({
                'hasEyeFlu': bool(prediction[0][0] > 0.5),
                'confidence': float(prediction[0][0])
            })
        
        except Exception as model_error:
            # Remove temp file if model processing fails
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Model processing error: {str(model_error)}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)