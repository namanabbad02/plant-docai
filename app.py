# # from fastapi import FastAPI, File, UploadFile, HTTPException
# # from fastapi.responses import JSONResponse
# # import tensorflow as tf
# # import numpy as np
# # from PIL import Image
# # import io
# # import os
# # import json # To save/load class names if not using image_dataset_from_directory metadata

# # # --- Configuration ---
# # # Specify which trained model to load ('color', 'grayscale', or 'segmented')
# # MODEL_IMAGE_TYPE = 'color' # <-- CHANGE THIS to load a different model
# # MODEL_PATH = f'best_model_color_phase2.h5'

# # # Model input dimensions (must match training)
# # IMG_WIDTH = 224
# # IMG_HEIGHT = 224

# # # Define your 38 class names in the correct order
# # # You can get this list from train_ds.class_names after running the training script,
# # # or if you saved them during training. Ensure the order is correct!
# # # Example (replace with your actual class names):
# # # CLASS_NAMES = [
# # #     'Apple scab', 'Black rot', 'Cedar apple rust', 'Cherry powdery mildew',
# # #     # ... add all 38 class names in alphabetical order as inferred by image_dataset_from_directory
# # # ]
# # # --- IMPORTANT: If you used image_dataset_from_directory, the class names are
# # #                inferred from the directory names in alphabetical order.
# # #                You need this exact list and order.
# # #                A good practice is to save train_ds.class_names after training.
# # #                For now, let's assume they are in the same order as your folders' alphabetical names.
# # #                Alternatively, you can load them from the dataset directory structure:
# # def get_class_names(dataset_base_dir='plants/plants'):
# #     try:
# #         train_dir = os.path.join(dataset_base_dir, 'train', MODEL_IMAGE_TYPE)
# #         if not os.path.exists(train_dir):
# #              raise FileNotFoundError(f"Training directory not found: {train_dir}")
# #         class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
# #         if not class_names:
# #              raise ValueError(f"No class folders found in {train_dir}")
# #         print(f"Inferred class names: {class_names}")
# #         return class_names
# #     except Exception as e:
# #         print(f"Error inferring class names: {e}. Please manually define CLASS_NAMES.")
# #         # Fallback to manual list or raise error
# #         raise SystemExit(f"Could not infer class names. Please ensure your 'plants/train/{MODEL_IMAGE_TYPE}' directory exists and contains class folders.")

# # CLASS_NAMES = get_class_names() # Try to infer from directory structure

# # # --- FastAPI App Instance ---
# # app = FastAPI(
# #     title="Plant Disease Detection API",
# #     description=f"API for predicting plant diseases using the {MODEL_IMAGE_TYPE} model.",
# #     version="1.0.0",
# # )

# # # --- Model and Class Names (Loaded on Startup) ---
# # model = None
# # # class_names_list = None # Using the globally inferred CLASS_NAMES instead

# # # --- Startup Event: Load the Model ---
# # @app.on_event("startup")
# # async def load_model():
# #     """Load the model when the FastAPI application starts."""
# #     global model
# #     print(f"Loading model from: {MODEL_PATH}")
# #     if not os.path.exists(MODEL_PATH):
# #         print(f"Error: Model file not found at {MODEL_PATH}")
# #         raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# #     try:
# #         # Custom objects might be needed if you used custom layers or metrics
# #         # For standard Keras layers and applications like ResNet50, this is often not needed
# #         model = tf.keras.models.load_model(MODEL_PATH)
# #         print("Model loaded successfully!")
# #         # Verify the model output shape matches the number of classes
# #         if model.output_shape[-1] != len(CLASS_NAMES):
# #              print(f"Warning: Model output shape ({model.output_shape[-1]}) does not match the number of inferred class names ({len(CLASS_NAMES)}).")
# #              print("Ensure MODEL_IMAGE_TYPE and CLASS_NAMES are correct.")

# #     except Exception as e:
# #         print(f"Error loading model: {e}")
# #         # Depending on the error, you might want to stop the app
# #         raise SystemExit(f"Failed to load the model from {MODEL_PATH}. Error: {e}")


# # # --- Prediction Endpoint ---
# # @app.post("/predict/")
# # async def predict_image(file: UploadFile = File(...)):
# #     """
# #     Predict the class of an uploaded image.

# #     Args:
# #         file: The image file to predict on.

# #     Returns:
# #         A JSON response with the predicted class and confidence scores.
# #     """
# #     if model is None:
# #         raise HTTPException(status_code=500, detail="Model not loaded. API is not ready.")

# #     # --- Image Preprocessing ---
# #     try:
# #         # Read the image file bytes
# #         image_bytes = await file.read()

# #         # Open the image using PIL
# #         img = Image.open(io.BytesIO(image_bytes))

# #         # Convert to RGB if necessary (ResNet expects 3 channels)
# #         if img.mode != 'RGB':
# #             img = img.convert('RGB')

# #         # Resize the image
# #         img = img.resize((IMG_WIDTH, IMG_HEIGHT))

# #         # Convert the image to a numpy array
# #         img_array = np.array(img)

# #         # Expand dimensions to create a batch size of 1
# #         # The model expects input in the shape (batch_size, height, width, channels)
# #         img_array = np.expand_dims(img_array, axis=0)

# #         # Apply the same preprocessing used during training (ResNet specific)
# #         # This layer was likely part of your Keras model pipeline
# #         # If you didn't include it in the saved model, apply it manually here:
# #         # img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
# #         # However, since we added it *after* the Input layer in the training script,
# #         # it should be part of the loaded model and applied automatically when model(inputs) is called.

# #     except Exception as e:
# #         print(f"Error during image preprocessing: {e}")
# #         raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

# #     # --- Make Prediction ---
# #     try:
# #         # Predict probabilities for each class
# #         predictions = model.predict(img_array) # This is a numpy array of probabilities

# #         # Get the class index with the highest probability
# #         predicted_class_index = np.argmax(predictions[0])

# #         # Get the confidence score for the predicted class
# #         confidence = float(predictions[0][predicted_class_index])

# #         # Get the predicted class name
# #         predicted_class_name = CLASS_NAMES[predicted_class_index]

# #         # Format all class probabilities
# #         confidence_scores = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}

# #     except Exception as e:
# #         print(f"Error during prediction: {e}")
# #         raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

# #     # --- Return Response ---
# #     return JSONResponse(content={
# #         "filename": file.filename,
# #         "predicted_class": predicted_class_name,
# #         "confidence": confidence,
# #         "all_class_confidences": confidence_scores
# #     })

# # # You can add a root endpoint for testing API status
# # @app.get("/")
# # async def read_root():
# #     return {"message": "Plant Disease Detection API is running."}

# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import JSONResponse
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import io
# import os
# import json

# # --- Configuration ---
# # Specify which trained model to load ('color', 'grayscale', or 'segmented')
# MODEL_IMAGE_TYPE = 'color' # <-- CHANGE THIS to load a different model
# # MODEL_PATH = f'best_model_{MODEL_IMAGE_TYPE}_phase2.h5'
# MODEL_PATH = f'best_model_{MODEL_IMAGE_TYPE}_phase2.h5'

# # Model input dimensions (must match training)
# IMG_WIDTH = 224
# IMG_HEIGHT = 224

# # Define your 38 class names in the correct order.
# # IMPORTANT: This list MUST match the order inferred by tf.keras.utils.image_dataset_from_directory
# # when you trained the model. It's usually alphabetical order of the directory names.
# # We'll try to infer them from the directory structure first, but you can manually
# # define CLASS_NAMES here if inference fails or isn't reliable.
# CLASS_NAMES = []

# # --- FastAPI App Instance ---
# app = FastAPI(
#     title="Plant Disease Detection API",
#     description=f"API for predicting plant diseases using the {MODEL_IMAGE_TYPE} model.",
#     version="1.0.0",
# )

# # --- Model and Class Names (Loaded on Startup) ---
# model = None

# # Function to infer class names from the training directory structure
# def infer_class_names(dataset_base_dir='plants/plants'):
#     """Infers class names from the sorted list of directory names in the training folder."""
#     try:
#         train_dir = os.path.join(dataset_base_dir, 'train', MODEL_IMAGE_TYPE)
#         if not os.path.exists(train_dir):
#              print(f"Training directory not found for class inference: {train_dir}")
#              return None
#         # List directories and sort them to match image_dataset_from_directory's behavior
#         class_dirs = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
#         if not class_dirs:
#              print(f"No class folders found in {train_dir} for inference.")
#              return None
#         print(f"Inferred class names (sorted): {class_dirs}")
#         return class_dirs
#     except Exception as e:
#         print(f"Error during class name inference: {e}")
#         return None

# # Function to determine the disease type string based on the class name
# def get_disease_type_from_class_name(class_name: str) -> str:
#     """
#     Extracts the disease type string from the full class name.
#     Assumes format like 'Plant___DiseaseName' or 'Plant___healthy'.
#     Replaces underscores with spaces.
#     """
#     if 'healthy' in class_name.lower():
#         return 'healthy' # Special case for healthy plants

#     parts = class_name.split('___')
#     if len(parts) > 1:
#         # Take the part after the '___' and replace underscores with spaces
#         disease_part = parts[1].replace('_', ' ')
#         # Optionally, refine common patterns like "spot/Bacterial spot" to just "Bacterial spot"
#         # This depends on your desired output granularity.
#         # For example, if "spot/Bacterial spot" is a class name, and you want the output "Bacterial spot":
#         if 'spot/' in disease_part:
#              return disease_part.split('/')[-1] # Takes the part after '/'
#         return disease_part
#     else:
#         # Handle cases that don't follow the Plant___Disease format
#         return class_name.replace('_', ' ') # Just replace underscores

# # --- Startup Event: Load the Model and Class Names ---
# @app.on_event("startup")
# async def load_resources():
#     """Load the model and infer class names when the FastAPI application starts."""
#     global model, CLASS_NAMES

#     # 1. Infer Class Names
#     inferred_names = infer_class_names()
#     if inferred_names:
#         CLASS_NAMES = inferred_names
#         print(f"Successfully inferred {len(CLASS_NAMES)} class names.")
#     else:
#         # Fallback: Manually define CLASS_NAMES here if inference failed
#         # Example (uncomment and populate if needed):
#         # CLASS_NAMES = [
#         #     'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
#         #     'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
#         #     'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
#         #     'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
#         #     'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
#         #     'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
#         #     'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
#         #     'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
#         #     'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
#         #     'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
#         #     'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
#         #     'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
#         #     'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
#         #     'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
#         #     'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
#         # ]
#         # print("Warning: Could not infer class names from directories. Using hardcoded list (if provided).")
#         if not CLASS_NAMES:
#              raise SystemExit(f"Could not determine class names. Please ensure '{os.path.join('plants', 'train', MODEL_IMAGE_TYPE)}' exists, contains class folders, or manually define CLASS_NAMES in the script.")

#     # 2. Load the Model
#     print(f"Loading model from: {MODEL_PATH}")
#     if not os.path.exists(MODEL_PATH):
#         print(f"Error: Model file not found at {MODEL_PATH}")
#         raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

#     try:
#         # Use MirroredStrategy if GPU is available for consistent loading
#         gpus = tf.config.list_physical_devices('GPU')
#         if gpus:
#              strategy = tf.distribute.MirroredStrategy()
#              with strategy.scope():
#                   model = tf.keras.models.load_model(MODEL_PATH)
#         else:
#              model = tf.keras.models.load_model(MODEL_PATH)

#         print("Model loaded successfully!")
#         # Verify the model output shape matches the number of classes
#         if model.output_shape[-1] != len(CLASS_NAMES):
#              print(f"Warning: Model output shape ({model.output_shape[-1]}) does not match the number of detected class names ({len(CLASS_NAMES)}).")
#              print("Ensure MODEL_IMAGE_TYPE is correct and the class names inference/list matches the trained model.")
#              # This might indicate a problem if the model was trained on a different class set

#     except Exception as e:
#         print(f"Error loading model: {e}")
#         raise SystemExit(f"Failed to load the model from {MODEL_PATH}. Error: {e}")


# # --- Prediction Endpoint ---
# @app.post("/predict/")
# async def predict_image(file: UploadFile = File(...)):
#     """
#     Predict the class and disease type of an uploaded image.

#     Args:
#         file: The image file to predict on.

#     Returns:
#         A JSON response with the predicted class, disease type, and confidence scores.
#     """
#     if model is None or not CLASS_NAMES:
#         raise HTTPException(status_code=500, detail="Model or Class Names not loaded. API is not ready.")

#     # --- Image Preprocessing ---
#     try:
#         # Read the image file bytes
#         image_bytes = await file.read()

#         # Open the image using PIL
#         img = Image.open(io.BytesIO(image_bytes))

#         # Convert to RGB if necessary (ResNet expects 3 channels)
#         if img.mode != 'RGB':
#             img = img.convert('RGB')

#         # Resize the image
#         img = img.resize((IMG_WIDTH, IMG_HEIGHT))

#         # Convert the image to a numpy array
#         img_array = np.array(img)

#         # Expand dimensions to create a batch size of 1
#         # The model expects input in the shape (batch_size, height, width, channels)
#         img_array = np.expand_dims(img_array, axis=0)

#         # The ResNet preprocessing layer is included in the saved model pipeline
#         # if you built it as shown in the training script. It will be applied automatically
#         # when the model makes a prediction.

#     except Exception as e:
#         print(f"Error during image preprocessing for {file.filename}: {e}")
#         raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

#     # --- Make Prediction ---
#     try:
#         # Predict probabilities for each class
#         predictions = model.predict(img_array) # This is a numpy array of probabilities (shape 1, NUM_CLASSES)

#         # Get the class index with the highest probability
#         predicted_class_index = np.argmax(predictions[0])

#         # Get the confidence score for the predicted class
#         confidence = float(predictions[0][predicted_class_index]) # Convert numpy float to standard float

#         # Get the predicted class name
#         predicted_class_name = CLASS_NAMES[predicted_class_index]

#         # Determine the broader disease type string
#         disease_type = get_disease_type_from_class_name(predicted_class_name)


#         # Format all class probabilities into a dictionary
#         confidence_scores = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}

#     except Exception as e:
#         print(f"Error during prediction for {file.filename}: {e}")
#         raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

#     # --- Return Response ---
#     return JSONResponse(content={
#         "filename": file.filename,
#         "predicted_class": predicted_class_name,
#         "confidence": confidence,
#         "disease_type": disease_type, # Add the new field
#         "all_class_confidences": confidence_scores
#     })

# # You can add a root endpoint for testing API status
# @app.get("/")
# async def read_root():
#     return {"message": "Plant Disease Detection API is running."}

# --- START OF FILE app.py ---

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware # Import CORS middleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import json # To save/load class names if not using image_dataset_from_directory metadata
# Note: json is not strictly needed with the current class name inference but kept if you add manual load/save

# --- Configuration ---
# Specify which trained model to load ('color', 'grayscale', or 'segmented')
# The API will load ONLY this model instance on startup.
# To use a different model type, you need to change this and restart the API.
MODEL_IMAGE_TYPE = 'color' # <-- CHANGE THIS to load a different model type
# Assuming the model is saved in the same directory as this app.py file
MODEL_PATH = f'best_model_{MODEL_IMAGE_TYPE}_phase2.h5'


# Model input dimensions (must match training)
IMG_WIDTH = 224
IMG_HEIGHT = 224

# Define your 38 class names in the correct order.
# IMPORTANT: This list MUST match the order inferred by tf.keras.utils.image_dataset_from_directory
# when you trained the model. It's usually alphabetical order of the directory names
# in the 'train/{MODEL_IMAGE_TYPE}' folder.
# We'll try to infer them from the directory structure first.
CLASS_NAMES = [] # Initialize as empty list

# --- FastAPI App Instance ---
app = FastAPI(
    title="Plant Disease Detection API",
    description=f"API for predicting plant diseases using the {MODEL_IMAGE_TYPE} model.",
    version="1.0.0",
)

# --- CORS Middleware ---
# Allows requests from your frontend (e.g., running on localhost:3000)
# In production, replace "*" with the specific domain(s) of your frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Be specific in production, e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"], # Allows GET, POST, etc.
    allow_headers=["*"], # Allows all headers
)


# --- Model and Class Names (Loaded on Startup) ---
model = None # Variable to hold the loaded TensorFlow model

# Function to infer class names from the training directory structure
# This helps ensure the CLASS_NAMES list matches the model's output order
def infer_class_names(dataset_base_dir='plants/plants'):
    """
    Infers class names from the sorted list of directory names
    in the training folder for the selected MODEL_IMAGE_TYPE.
    Assumes 'plants' is relative to the directory where app.py is run.
    """
    try:
        # Construct the path to the training directory for the selected image type
        # Adjust 'plants' if your dataset structure is nested differently
        train_dir = os.path.join(dataset_base_dir, 'train', MODEL_IMAGE_TYPE)
        print(f"Attempting to infer class names from: {train_dir}")
        if not os.path.exists(train_dir):
             print(f"Training directory not found for class inference: {train_dir}")
             # If directory doesn't exist, we cannot infer. Return None.
             return None
        # List directories within the train_dir and sort them.
        # image_dataset_from_directory sorts class names alphabetically.
        class_dirs = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        if not class_dirs:
             print(f"No class folders found in {train_dir} for inference.")
             return None
        print(f"Inferred {len(class_dirs)} class names (sorted): {class_dirs}")
        return class_dirs
    except Exception as e:
        print(f"Error during class name inference: {e}")
        return None

# Function to determine the disease type string based on the class name
# This logic is moved from model.ts to the backend
def get_disease_type_from_class_name(class_name: str) -> str:
    """
    Extracts the disease type string from the full class name.
    Assumes format like 'Plant___DiseaseName' or 'Plant___healthy'.
    Replaces underscores with spaces.
    """
    # Check for the common 'healthy' class pattern first
    if 'healthy' in class_name.lower():
        return 'Healthy' # Standardize 'healthy' capitalization

    parts = class_name.split('___')
    if len(parts) > 1:
        # Take the part after the '___'
        disease_part = parts[1]
        # Replace underscores with spaces
        disease_part = disease_part.replace('_', ' ')
        # Optional refinement for common patterns like "spot/Bacterial spot"
        # This depends on your desired output format.
        if '/' in disease_part:
             return disease_part.split('/')[-1] # Takes the part after the last '/'
        return disease_part
    else:
        # Handle cases that don't follow the Plant___Disease format
        # Or if the split doesn't yield two parts (e.g., just a name)
        return class_name.replace('_', ' ') # Just replace underscores

# Treatment Map - Moved from model.ts to the backend
# This map links class names or disease types to recommended treatments
# Using the full class name as key is safer as it's unique
TREATMENT_MAP: dict[str, str] = {
    'Apple___Apple_scab': 'Apply fungicide containing sulfur or copper. Prune affected branches.',
    'Apple___Black_rot': 'Remove infected fruit and branches. Apply fungicides at early stages.',
    'Apple___Cedar_apple_rust': 'Remove cedar trees within a few miles or prune galls. Apply fungicides.',
    'Apple___healthy': 'Continue regular care: watering, fertilization, and pruning.',
    'Blueberry___healthy': 'Continue regular care: watering, fertilization, and pruning.',
    'Cherry_(including_sour)___Powdery_mildew': 'Apply fungicides like sulfur or potassium bicarbonate. Improve air circulation.',
    'Cherry_(including_sour)___healthy': 'Continue regular care: watering, fertilization, and pruning.',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Use resistant varieties. Rotate crops. Apply fungicides if severe.',
    'Corn_(maize)___Common_rust_': 'Use resistant varieties. Apply fungicides early in the season if disease is prevalent.',
    'Corn_(maize)___Northern_Leaf_Blight': 'Use resistant varieties. Rotate crops. Apply fungicides.',
    'Corn_(maize)___healthy': 'Continue regular care.',
    'Grape___Black_rot': 'Apply fungicides throughout the growing season. Prune and destroy infected parts.',
    'Grape___Esca_(Black_Measles)': 'Prune out symptomatic wood. Improve vine health.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Apply fungicides. Remove infected leaves.',
    'Grape___healthy': 'Continue regular care.',
    'Orange___Haunglongbing_(Citrus_greening)': 'No cure. Manage psyllid vector with insecticides. Remove infected trees to prevent spread.',
    'Peach___Bacterial_spot': 'Use resistant varieties. Apply copper sprays. Prune infected branches.',
    'Peach___healthy': 'Continue regular care.',
    'Pepper,_bell___Bacterial_spot': 'Use resistant seeds/varieties. Apply copper sprays. Avoid overhead watering.',
    'Pepper,_bell___healthy': 'Continue regular care.',
    'Potato___Early_blight': 'Rotate crops. Avoid overhead irrigation. Apply fungicides.',
    'Potato___Late_blight': 'Plant resistant varieties. Use fungicides preventatively. Ensure good air circulation.',
    'Potato___healthy': 'Continue regular care.',
    'Raspberry___healthy': 'Continue regular care.',
    'Soybean___healthy': 'Continue regular care.',
    'Squash___Powdery_mildew': 'Apply fungicides (sulfur, neem oil, potassium bicarbonate). Ensure good air circulation.',
    'Strawberry___Leaf_scorch': 'Remove infected leaves. Use fungicides. Ensure good drainage.',
    'Strawberry___healthy': 'Continue regular care.',
    'Tomato___Bacterial_spot': 'Use certified disease-free seeds/plants. Apply copper sprays. Remove infected plants.',
    'Tomato___Early_blight': 'Rotate crops. Mulch to prevent soil splashing. Apply fungicides.',
    'Tomato___Late_blight': 'Use resistant varieties. Apply fungicides preventatively. Ensure good air circulation.',
    'Tomato___Leaf_Mold': 'Improve ventilation. Avoid overhead watering. Apply fungicides if needed.',
    'Tomato___Septoria_leaf_spot': 'Remove lower leaves. Mulch. Rotate crops. Apply fungicides.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Rinse plants with water. Use insecticidal soap or miticides. Introduce beneficial insects.',
    'Tomato___Target_Spot': 'Rotate crops. Use fungicides. Avoid overhead watering.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Control whiteflies (vector) with insecticides and reflective mulches. Remove infected plants.',
    'Tomato___Tomato_mosaic_virus': 'No cure. Remove infected plants. Sanitize tools. Avoid tobacco products around plants.',
    'Tomato___healthy': 'Continue regular care.'
    # ...add the rest of your 38 treatments
}


# --- Startup Event: Load the Model and Class Names ---
@app.on_event("startup")
async def load_resources():
    """
    Load the model and infer class names when the FastAPI application starts.
    """
    global model, CLASS_NAMES

    # 1. Infer Class Names from the training directory structure
    inferred_names = infer_class_names()
    if inferred_names:
        CLASS_NAMES = inferred_names
        print(f"Successfully inferred {len(CLASS_NAMES)} class names.")
    else:
        # If inference failed, you *must* manually define CLASS_NAMES here
        # Otherwise, the API cannot start because it doesn't know the classes.
        # Uncomment and populate the following list if inference fails:
        # CLASS_NAMES = [
            # 'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
            # 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
            # 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            # 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
            # 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
            # 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            # 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
            # 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
            # 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            # 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
            # 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
            # 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
            # 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
            # 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            # 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
        # ]
        # print("Warning: Could not infer class names from directories. Using hardcoded list (if available).")
        if not CLASS_NAMES:
             print("Critical Error: Class names could not be inferred and no manual list was provided.")
             print(f"Please ensure '{os.path.join('plants', 'train', MODEL_IMAGE_TYPE)}' exists and contains class folders,")
             print("or manually define the CLASS_NAMES list in app.py.")
             raise SystemExit("Failed to determine class names.")


    # 2. Load the Model
    print(f"Loading model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        # Exit the application if the model file is missing
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    try:
        # Use MirroredStrategy if GPU is available for consistent loading,
        # although for simple loading it might not be strictly necessary,
        # it's good practice if the model was saved/trained with strategies.
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
             print(f"Detected GPUs: {len(gpus)}. Using MirroredStrategy for model loading.")
             strategy = tf.distribute.MirroredStrategy()
             with strategy.scope():
                  # Loading the model within the strategy scope
                  model = tf.keras.models.load_model(MODEL_PATH)
        else:
             print("No GPUs detected or configured. Loading model on CPU.")
             # Load the model normally if no GPU strategy is used
             model = tf.keras.models.load_model(MODEL_PATH)

        print("Model loaded successfully!")
        # Verify the model output shape matches the number of classes
        if model.output_shape[-1] != len(CLASS_NAMES):
             print(f"Warning: Model output shape ({model.output_shape[-1]}) does not match the number of detected class names ({len(CLASS_NAMES)}).")
             print("Ensure MODEL_IMAGE_TYPE configuration and class names inference/list match the trained model.")
             # Decide if this should be a fatal error. For now, it's a warning.

    except Exception as e:
        print(f"Error loading model: {e}")
        # Exit the application if model loading fails
        raise SystemExit(f"Failed to load the model from {MODEL_PATH}. Error: {e}")


# --- Prediction Endpoint ---
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict the class and disease type of an uploaded image using the loaded model.

    Args:
        file: The image file to predict on, uploaded as multipart/form-data.

    Returns:
        A JSON response with prediction details.
    """
    # Check if the model and class names were loaded successfully on startup
    if model is None or not CLASS_NAMES or len(CLASS_NAMES) != model.output_shape[-1]:
        # This indicates a startup failure
        print("Error: Model or Class Names not properly initialized.")
        raise HTTPException(status_code=500, detail="Server is not ready: Model or Class Names not loaded.")

    # --- Image Preprocessing ---
    try:
        # Read the image file bytes asynchronously
        image_bytes = await file.read()

        # Open the image using PIL from the bytes
        img = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if necessary (most models expect 3 color channels)
        # PIL's convert handles various input formats ('L', 'RGBA', etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize the image to the dimensions expected by the model
        # Use Image.Resampling.NEAREST or other methods if needed, but BILINEAR is common
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.BILINEAR)

        # Convert the PIL image object to a numpy array
        img_array = np.array(img)

        # Expand dimensions to create a batch size of 1.
        # The model expects input in the shape (batch_size, height, width, channels)
        img_array = np.expand_dims(img_array, axis=0)

        # --- Model-Specific Preprocessing ---
        # If your training pipeline included a `preprocess_input` layer (like for ResNet50)
        # added *after* the initial Input layer but *before* the base_model,
        # it should be part of the loaded model graph and applied automatically
        # when you call model.predict().
        # If you applied preprocessing *before* feeding to the base model in training
        # and did NOT include it in the saved model, you would apply it here manually:
        # img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        # Based on the previous training script structure, it's likely included
        # in the model itself, so we don't need to call it explicitly here.

    except Exception as e:
        print(f"Error during image preprocessing for {file.filename}: {e}")
        # Return a 400 Bad Request error if image processing fails
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

    # --- Make Prediction ---
    try:
        # Use the loaded model to predict probabilities.
        # This runs on the GPU if available and configured.
        predictions = model.predict(img_array) # Output shape (1, NUM_CLASSES)

        # The output is a numpy array. Get the probabilities for the single image (index 0).
        probabilities = predictions[0] # Shape (NUM_CLASSES,)

        # Find the index of the class with the highest probability
        predicted_class_index = np.argmax(probabilities)

        # Get the confidence score (probability) for the predicted class
        confidence = float(probabilities[predicted_class_index]) # Convert numpy float to standard Python float

        # Get the predicted class name using the index
        # predicted_class_name = CLASS_NAMES[predicted_class_index]
        full_class_name = CLASS_NAMES[predicted_class_index]
        
        predicted_class_name = full_class_name.split("_")[0]

        # Determine the broader disease type string
        disease_type = get_disease_type_from_class_name(full_class_name)

        # Get the recommended treatment using the full predicted class name
        # Fallback to a default message if the class is not in the map
        treatment = TREATMENT_MAP.get(full_class_name, "No specific treatment found for this diagnosis.")


        # Format all class probabilities into a dictionary {class_name: probability}
        # Convert numpy floats to standard Python floats for JSON serialization
        all_class_confidences = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}

    except Exception as e:
        print(f"Error during prediction for {file.filename}: {e}")
        # Return a 500 Internal Server Error if prediction fails
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

    # --- Return Response ---
    # Return a JSON response containing all the relevant prediction details
    return JSONResponse(content={
        "filename": file.filename, # Include original filename for reference
        "predicted_class": full_class_name, # The full class name (e.g., 'Tomato___Late_blight')
        "disease_type": disease_type, # The extracted disease name (e.g., 'Late blight')
        "confidence": confidence, # Confidence score for the predicted class
        "treatment": treatment, # Recommended treatment
        "scores": all_class_confidences, # Dictionary of all class scores
    })

# You can add a root endpoint for testing API status
@app.get("/")
async def read_root():
    """Basic endpoint to check if the API is running."""
    status = "API is running"
    if model is None:
        status += " - Model is NOT loaded"
    else:
         status += " - Model is loaded"
    if not CLASS_NAMES:
         status += " - Class Names NOT loaded"
    else:
         status += f" - {len(CLASS_NAMES)} Class Names loaded"

    return {"message": status}

# --- END OF FILE app.py ---