import * as tf from '@tensorflow/tfjs';

export interface PredictionResult {
  disease: string;
  confidence: number;
  treatment: string;
}

const DISEASES = {
  healthy: "No disease detected",
  leaf_blight: "Leaf Blight",
  powdery_mildew: "Powdery Mildew",
  // Add more diseases as needed
};

const TREATMENTS = {
  healthy: "No treatment needed. Continue regular plant care.",
  leaf_blight: "Apply copper-based fungicide and ensure proper air circulation.",
  powdery_mildew: "Apply sulfur-based fungicide and reduce humidity.",
  // Add more treatments as needed
};

let model: tf.LayersModel | null = null;

export async function loadModel() {
  try {
    // Replace with your model URL
    model = await tf.loadLayersModel('/model/model.json');
    console.log('Model loaded successfully');
  } catch (error) {
    console.error('Error loading model:', error);
    throw error;
  }
}

export async function preprocessImage(imageData: string): Promise<tf.Tensor> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      // Resize and normalize image
      const tensor = tf.tidy(() => {
        const imageTensor = tf.browser.fromPixels(img)
          .resizeNearestNeighbor([224, 224]) // Resize to model input size
          .toFloat()
          .expandDims();
        return imageTensor.div(255.0); // Normalize to [0,1]
      });
      resolve(tensor);
    };
    img.onerror = (error) => reject(error);
    img.src = imageData;
  });
}

export async function predict(imageData: string): Promise<PredictionResult> {
  if (!model) {
    throw new Error('Model not loaded');
  }

  const tensor = await preprocessImage(imageData);
  const predictions = await model.predict(tensor) as tf.Tensor;
  const probabilities = await predictions.data();
  
  // Cleanup
  tensor.dispose();
  predictions.dispose();

  // Get highest probability class
  const maxProbIndex = probabilities.indexOf(Math.max(...Array.from(probabilities)));
  const diseaseKey = Object.keys(DISEASES)[maxProbIndex];
  
  return {
    disease: DISEASES[diseaseKey],
    confidence: probabilities[maxProbIndex],
    treatment: TREATMENTS[diseaseKey]
  };
}