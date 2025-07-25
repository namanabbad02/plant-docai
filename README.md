# MONOTARS - AI-Powered Plant Disease Detection System with Multilingual Chatbot 🌿🧠

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![React](https://img.shields.io/badge/React-18%2B-61DAFB?logo=react)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-4.x-3178C6?logo=typescript)](https://www.typescriptlang.org/)
[![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-3.x-06B6D4?logo=tailwindcss)](https://tailwindcss.com/)

## 📝 Table of Contents

*   [Project Overview](#-project-overview)
*   [Key Features](#-key-features)
*   [System Architecture](#-system-architecture)
*   [Tech Stack](#-tech-stack)
*   [Model Details](#-model-details)
*   [Dataset](#-dataset)
*   [Getting Started](#-getting-started)
    *   [Prerequisites](#prerequisites)
    *   [1. Data Preparation](#1-data-preparation)
    *   [2. Model Training](#2-model-training)
    *   [3. Model Conversion for Web](#3-model-conversion-for-web)
    *   [4. Frontend Application Setup](#4-frontend-application-setup)
    *   [5. Running the Application](#5-running-the-application)
    *   [6. (Optional) Running the Local FastAPI for Server-Side Testing](#6-optional-running-the-local-fastapi-for-server-side-testing)
*   [Usage](#-usage)
*   [Future Enhancements](#-future-enhancements)
*   [Team & Guidance](#-team--guidance)
*   [License](#-license)
*   [Contact](#-contact)

## ✨ Project Overview

**MONOTARS - Growing for Less** is an intelligent, AI-powered plant disease detection system designed to assist farmers and enthusiasts in identifying crop diseases at an early stage through image-based analysis. This comprehensive system leverages cutting-edge deep learning, computer vision, and integrates real-time environmental data for a holistic approach to crop monitoring and management.

The project addresses the challenges of traditional, often manual, disease identification by providing a scalable, low-cost, and user-friendly solution. It features a responsive web frontend with an intelligent multilingual chatbot, image upload capabilities for immediate diagnosis, and a robust backend infrastructure (though for the frontend, the core image inference is performed client-side using TensorFlow.js for speed and efficiency). MONOTARS aims to democratize access to AI in agriculture, promote sustainable farming practices, and contribute to global food security.

## 🚀 Key Features

*   **AI-Powered Disease Diagnosis:** Accurate identification of plant diseases from uploaded images using a hybrid deep learning model.
*   **38 Disease/Healthy Classes:** Classifies images into 38 distinct categories covering various plant species and their diseases (e.g., fungal, bacterial, viral infections, pest damage) and healthy states.
*   **Hybrid Deep Learning Model (CNN + ViT):** Utilizes a sophisticated hybrid deep learning model combining Convolutional Neural Networks (CNNs) for global feature extraction and Vision Transformers (ViTs) for localized attention, enhancing prediction accuracy under diverse visual conditions.
*   **Detailed & Actionable Insights:** Provides not only the specific predicted class and confidence but also a simplified "Disease Type" and practical treatment recommendations.
*   **Multilingual Chatbot Interface:** An interactive chatbot with planned speech-to-text and text-to-speech functionalities, supporting multiple languages (English, Hindi, Kannada) for broad accessibility.
*   **Client-Side Inference:** Core image prediction is executed directly within the browser using TensorFlow.js, enabling real-time feedback and reducing server load.
*   **IoT-Based Monitoring (Planned/Architectural):** Designed to integrate with low-cost IoT modules (e.g., Raspberry Pi with sensors) for real-time environmental data collection, enhancing context-aware diagnosis.
*   **Federated Learning (Planned/Architectural):** Incorporates Federated Learning for continual model improvement and adaptability without centralizing sensitive user data, preserving privacy.
*   **Edge Deployment Capability:** Architected for scalability and low-cost deployment on edge devices like Raspberry Pi, making the solution practical for rural areas with limited internet connectivity.
*   **User-Friendly Frontend:** Intuitive interface with drag-and-drop image upload and seamless chatbot interaction.

## 📐 System Architecture

The MONOTARS system is a full-stack application envisioned with a hybrid AI-IoT approach. The simplified diagram below illustrates the core interactions:
```
        +-----------------------------+
        |         User Interface      |
        |   (React + Tailwind + Chat) |
        +-------------+---------------+
                      |
     Upload Image     |   Ask Query
                      ↓
   +---------------------+       +-------------------------+
   |  TensorFlow.js Model| <---> |        FastAPI           |
   |  (ResNet50+ViT .json)|      |   (optional for Chat)    |
   +---------------------+       +-------------------------+

```
**Explanation of Flow:**
*   **Image Prediction:** Users `Upload Image` directly to the **TensorFlow.js Model** running client-side within the browser. The prediction results are then displayed in the **User Interface**.
*   **Text Chat:** User `Ask Query` from the **User Interface** is sent to the **Supabase/FastAPI** backend (Supabase for text-based chat in the current implementation). Replies are sent back from the backend to the UI.
*   The connection between the **TensorFlow.js Model** and **Supabase/FastAPI** in the diagram signifies that while the image model runs in the browser, the overall project envisions a comprehensive system where the backend could provide additional services related to model management, data storage, or other AI tasks.

**Components Breakdown:**

*   **Frontend:** A responsive web interface built with React and TypeScript, handling user interaction, image uploads, displaying predictions, and integrating multilingual chat.
*   **Backend (AI Prediction API - *Architectural*):** A FastAPI-based Python backend designed for AI-based image classification and data flow. While the primary image inference for the *current frontend code* is client-side, this backend component is integral to the overall system's robust architecture and for server-side deployments/testing.
*   **Backend (Text Chat - *External*):** Supabase Functions are used for handling the text-based queries to the chatbot.
*   **IoT Module (Future/Architectural):** Raspberry Pi and sensors for environmental data collection.
*   **Hybrid Deep Learning Model:** Trained models (CNNs + ViTs) reside either on the edge device, a central backend, or are converted for client-side use.


## 🛠️ Tech Stack

This project uses a split architecture: Python for model training and a React/TypeScript frontend for the web application, leveraging TensorFlow.js for client-side AI inference.

**Machine Learning & Data Processing (Python - for Training & Optional Server-side Inference):**

*   **Language:** `Python 3.9+`
*   **Frameworks:** `TensorFlow 2.x / Keras` (for model building, training, loading), `Scikit-learn` (evaluation metrics).
*   **Utilities:** `NumPy` (numerical operations), `Pandas` (metadata handling), `Pillow` (PIL - image processing), `Matplotlib` / `Seaborn` (visualization), `os`, `shutil`, `random` (data segregation).
*   **Model Conversion:** `tensorflowjs_converter` (to prepare models for web deployment).
*   **API (Optional):** `FastAPI` (for building RESTful API endpoints for server-side inference testing/deployment), `Uvicorn` (ASGI server).
*   **Image Processing (for training):** `OpenCV` (for tasks like resizing, normalization, and augmentation).

**Frontend (React/TypeScript - Client-Side Application):**

*   **Framework:** `React 18+` (with `TypeScript 4.x`) for building interactive UI components.
*   **Bundler:** `Vite` (for fast development and builds).
*   **UI Libraries:** `Tailwind CSS` (for styling), `Lucide-react` (for icons).
*   **File Upload:** `react-dropzone` (for drag-and-drop plant image uploads).
*   **AI Inference:** `@tensorflow/tfjs` (for running the deep learning model directly in the browser).
*   **Chat UI:** Custom `ChatMessage.tsx` (for displaying chat messages), `LanguageSelector.tsx` with `LanguageContext` (for dynamic multilingual interface).
*   **API Communication:** `fetch` (for communicating with external services like Supabase for text chat).
*   **State Management:** `React Hooks` (`useState`, `useEffect`, `useRef`) for managing application state.

**Backend (External for Text Chat):**

*   **Platform:** Supabase Functions (for handling general text-based chatbot inquiries).

## 🧠 Model Details

*   **Core Architecture:** Hybrid Deep Learning Model combining:
    *   **Convolutional Neural Networks (CNNs):** Such as ResNet50, for robust feature extraction from images.
    *   **Vision Transformers (ViTs):** For enhanced performance in complex visual conditions, providing localized attention.
*   **Approach:** Transfer Learning with two phases:
    1.  **Phase 1 (Feature Extraction):** Utilizes pre-trained weights from ImageNet, with only a custom classification head trained on the specific plant disease dataset.
    2.  **Phase 2 (Fine-tuning):** Selectively unfreezes and trains later layers of the base model with a very low learning rate to adapt learned features more precisely to plant disease patterns.
*   **Input:** 224x224 pixels, 3 color channels (RGB).
*   **Preprocessing:** Images are resized, converted to floating-point numbers, and normalized (e.g., channel-wise mean subtraction, similar to `tf.keras.applications.resnet50.preprocess_input`). This preprocessing must be consistently applied both during Python training and TF.js inference.
*   **Output:** 38 distinct classes (one-hot encoded), with a softmax activation function outputting probability distributions for each class.

## 💾 Dataset

The project is trained on a comprehensive dataset, specifically the **PlantVillage dataset**.

*   **Size:** Contains over 54,000 labeled images.
*   **Content:** Represents 14 different crop species (e.g., tomatoes, peppers, apples) and a wide range of diseases (fungal, bacterial, and viral infections, as well as pest damage) as well as healthy instances.
*   **Organization:** Images are organized by crop type and specific disease, following a clear naming convention like "Crop___Disease" for straightforward identification and classification.
*   **Splitting:** The dataset is segregated into 80% for training and 20% for testing/validation to ensure robust model development and unbiased evaluation.

## 🚀 Getting Started

Follow these instructions to set up and run the MONOTARS project on your local machine.

### Prerequisites

*   **Python 3.9+** (with `pip` installed)
*   **Node.js (LTS version recommended)** and **npm** (or `yarn`)
*   **Git**

### 1. Data Preparation

First, you need to segregate your raw dataset into `train` and `test` directories with the specified structure.

1.  **Organize Raw Data:** Place your initial dataset structure in the `plants/` directory:
    ```
    plants/
    ├── color/
    │   ├── Class_A/ (images)
    │   ├── Class_B/ (images)
    │   └── ... (all 38 class folders)
    ├── grayscale/
    │   ├── Class_A/ (images)
    │   └── ...
    └── segmented/
        ├── Class_A/ (images)
        └── ...
    ```
2.  **Navigate:** Open your terminal and navigate to the root directory of this project:
    ```bash
    cd /path/to/your/project-root
    ```
3.  **Run Segregation Script:** Execute the Python script. This will **move** your image files, so consider backing up your original dataset first.
    ```bash
    python split_dataset.py
    ```
    This creates `plants/train` and `plants/test` directories, each containing `color`, `grayscale`, and `segmented` subfolders with an 80/20 split of your original images.

### 2. Model Training

After data segregation, train your machine learning model.

1.  **Install Python Dependencies:**
    ```bash
    pip install tensorflow scikit-learn numpy matplotlib Pillow fastapi uvicorn tensorflowjs opencv-python pandas
    ```
    *   For GPU Training: Ensure you have compatible NVIDIA drivers, CUDA Toolkit, and cuDNN installed for your TensorFlow version.
2.  **Run Training Script:** Execute your training script (e.g., `train_model.py`). This script should be configured to save your trained Keras model (e.g., as `best_model_color_phase2.h5`).
    ```bash
    python train_model.py
    ```

### 3. Model Conversion for Web

The trained Keras model (`.h5`) needs to be converted into a TensorFlow.js (`tfjs_layers_model`) format for use in the browser.

1.  **Ensure `tensorflowjs` is installed** (from step 2.1).
2.  **Create Model Directory:** Make sure the target directory exists within your React app's `public` folder:
    ```bash
    mkdir -p public/model
    ```
3.  **Convert Model:** Run the conversion command. **Replace `./best_model_color_phase2.h5` with the actual path and filename of your trained Keras model.**
    ```bash
    tensorflowjs_converter \
      --input_format=keras \
      --output_format=tfjs_layers_model \
      ./best_model_color_phase2.h5 \
      ./public/model
    ```
    This will generate `model.json` and several `.weights.bin` files inside the `public/model/` directory.

### 4. Frontend Application Setup

Set up the React frontend application.

1.  **Install Node.js Dependencies:**
    ```bash
    npm install
    # OR
    yarn install
    ```
2.  **Environment Variables (for Text Chat):** Create a `.env` file in your project's root directory for Supabase integration:
    ```dotenv
    VITE_SUPABASE_URL=YOUR_SUPABASE_PROJECT_URL
    VITE_SUPABASE_ANON_KEY=YOUR_SUPABASE_ANON_KEY
    ```

### 5. Running the Application

1.  **Start Development Server:**
    ```bash
    npm run dev
    # OR
    yarn dev
    ```
2.  Open your web browser and navigate to the address displayed in the terminal (e.g., `http://localhost:5173` or `http://localhost:3000`).
3.  The application will load, and the AI model will be loaded in the background from `public/model/`. The UI inputs will become active once the model loading is complete.

### 6. (Optional) Running the Local FastAPI for Server-Side Testing

If you want to test the Python FastAPI prediction endpoint locally (as part of the broader system architecture):

1.  Ensure `fastapi` and `uvicorn` are installed (from step 2.1).
2.  Navigate to your project root in the terminal.
3.  Run the FastAPI application:
    ```bash
    uvicorn api:app --reload
    ```
4.  Access the API documentation at `http://127.0.0.1:8000/docs`. You can upload images here to test the Python model directly.

## 💡 Usage

Once the application is running in your browser:

1.  **Image-Based Diagnosis:**
    *   On the left panel, click or drag an image file of a plant leaf (JPEG, JPG, PNG) into the designated upload area.
    *   The model will process the image client-side, and the chatbot will display the predicted disease class, type, confidence, and recommended treatment.
2.  **Text Chat:**
    *   Type your plant-related questions or messages into the input field at the bottom of the chat interface.
    *   Press `Enter` or click the `Send` button.
    *   The chatbot will respond to your queries (via the external Supabase Function).

## 🔮 Future Enhancements

As per the project's vision, future enhancements include:

*   **Explainable AI (XAI) Integration:** Implement methods to provide insights into model decisions, increasing user trust.
<!--*   **Drone-Based Surveillance:** Integrate with drones for large-scale, automated field monitoring and disease detection.-->
*   **Domain-Specific Knowledge Integration:** Enhance the chatbot's knowledge base with more nuanced agricultural information.
*   **Integration with Agricultural Advisories & Marketplaces:** Connect farmers with expert advice and relevant market information.
*   **Advanced Speech-to-Text & Text-to-Speech:** Further refine voice interaction capabilities for improved accessibility.
*   **Emotional Tone Detection:** Analyze user sentiment in chat interactions for more empathetic responses.

## 🧑‍💻 Team & Guidance

This project was developed as a Bachelor of Technology project in Computer Science and Engineering (AIML) at Jain (Deemed to be University) Bengaluru.

*   **Students:**
    *   Aryan Jha 
    *   Naman 
    *   Dhrub Kumar Jha 
    *   Rehan 
    *   Shaily Shah 
    *   Yash Raj Roy 
*   **Under the Guidance of:**
    *   Dr. S Kanithan, Assistant Professor
<!---*   **Project Period:** February, 2023 to April, 2024 -->

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For any questions or feedback, feel free to reach out:

[Naman/https://www.linkedin.com/in/naman-n-jain/] - [abbadnaman@gmail.com]
