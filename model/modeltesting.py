import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the best model
model_path = r'C:\Users\Ronit\Downloads\EyeVision\Dataset\best_model.keras'
model = load_model(model_path)
print("Model loaded successfully.")

# Function to preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = load_img(image_path, target_size=target_size)
    # Convert image to array
    img_array = img_to_array(img)
    # Normalize the image
    img_array = img_array / 255.0
    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict_eye_flu(image_path):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)
    # Predict using the model
    prediction = model.predict(preprocessed_image)
    # Interpret the result
    if prediction[0][0] > 0.5:
        print(f"The image at {image_path} is classified as having eye flu.")
    else:
        print(f"The image at {image_path} is classified as not having eye flu.")

# Path to the test images folder
test_dir = r'C:\Users\Ronit\Downloads\EyeVision\Dataset\test'

# List all files in the test folder
test_images = os.listdir(test_dir)

# Ensure there is at least one image in the folder
if test_images:
    # Select the first image (or randomly choose an image)
    test_image_path = os.path.join(test_dir, test_images[0])  # Change this if you want to pick a specific one
    print(f"Testing with image: {test_image_path}")
    # Predict the result
    predict_eye_flu(test_image_path)
else:
    print(f"No images found in the folder: {test_dir}")
