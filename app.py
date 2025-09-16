import gradio as gr
import numpy as np
from PIL import Image
import os
import requests
import base64
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from io import BytesIO

# Class names
CLASS_NAMES = ["Normal leaf", "Red rot", "White leaf"]

# Image preprocessing
def preprocess_image(img):
    """Preprocess the image for the model"""
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def create_mock_model():
    """Create a simple mock model for demo purposes"""
    class MockModel:
        def predict(self, x):
            # Return random predictions for demo
            return np.random.rand(1, 3)  # 3 classes
    return MockModel()

def load_model():
    """Try to load the model, fall back to mock if not available"""
    model_path = "sugercane1.keras"
    
    # Check if model file exists and is not empty
    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
        print(f"Model file {model_path} not found or is empty.")
        return create_mock_model(), True  # Return mock model
    
    try:
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Model loaded successfully!")
        return model, False  # Return real model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure the model file is not corrupted")
        print("2. Try re-saving the model with: model.save('sugercane1.keras')")
        print("3. Check TensorFlow version compatibility")
        return create_mock_model(), True  # Fall back to mock model

# Load the model when the app starts
print("Starting application...")
model, is_mock = load_model()
if is_mock:
    print("\nWARNING: Using mock model for predictions.")
    print("To use a real model, ensure you have a valid 'sugercane1.keras' file.\n")

def predict_disease(img):
    """Predict disease using the trained model or mock model"""
    try:
        # Preprocess the image
        processed_img = preprocess_image(img)
        
        # Get predictions
        predictions = model.predict(processed_img, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Get class name
        if is_mock:
            class_name = f"Demo: {CLASS_NAMES[predicted_class_idx]}"
            # Add some randomness to make it look more realistic
            confidence = max(0.7, min(0.99, confidence + np.random.uniform(-0.1, 0.1)))
        else:
            class_name = CLASS_NAMES[predicted_class_idx]
            
        return class_name, confidence
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return "Prediction Error", 0.0

# Function to get advice using Ollama
def get_llm_advice(diagnosis):
    """Get advice from LLM based on the diagnosis"""
    try:
        # Prepare the prompt
        prompt = f"""You are an expert in plant pathology. Provide specific care advice for a plant with the following condition: {diagnosis}.
        
        Your response should include:
        1. A brief description of the condition
        2. Recommended treatment steps
        3. Preventive measures
        4. When to consult a professional
        
        Be concise but thorough in your advice."""
        
        # Prepare the request to Ollama
        data = {
            "model": "gemma3",
            "prompt": prompt,
            "stream": False
        }
        
        # Make the request
        response = requests.post("http://localhost:11434/api/generate", json=data)
        response.raise_for_status()
        result = response.json()
        
        return result.get("response", "Could not generate advice at this time.")
        
    except Exception as e:
        print(f"Error getting advice: {str(e)}")
        return "Error generating advice. Please try again later."
    except Exception as e:
        return f"Error making prediction: {str(e)}"

def analyze_image(img):
    """Analyze image and return diagnosis and advice"""
    if model is None:
        return "Error: Model not loaded. Please check the logs.", ""
    
    try:
        # Get prediction from the model
        prediction, confidence = predict_disease(img)
        
        # Get advice from LLM
        advice = get_llm_advice(prediction)
        
        return f"{prediction} (Confidence: {confidence:.2f})", advice
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        return f"Error: {str(e)}", ""

# Create the Gradio interface
with gr.Blocks(title="Crop Health Advisor") as demo:
    gr.Markdown("""
    # 🌱 Crop Health Advisor
    Upload an image of your crop to get a diagnosis and care advice.
    """)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Crop Image")
            submit_btn = gr.Button("Analyze")
        
        with gr.Column():
            output_diagnosis = gr.Textbox(label="Diagnosis", interactive=False)
            output_advice = gr.Textbox(label="Care Advice", lines=4, interactive=False)
    
    # Chat interface for follow-up questions
    with gr.Accordion("Ask a follow-up question", open=False):
        chat_input = gr.Textbox(label="Your question", placeholder="Ask about treatment options...")
        chat_output = gr.Textbox(label="Response", interactive=False)
        chat_btn = gr.Button("Ask")
    
    # Define button actions
    submit_btn.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=[output_diagnosis, output_advice]
    )
    
    def respond_to_question(question, diagnosis):
        """Generate a response to the user's question using the LLM"""
        if not diagnosis:
            return "Please analyze an image first so I can provide relevant advice."
        
        try:
            # Get the base diagnosis without the confidence score
            base_diagnosis = diagnosis.split('(')[0].strip()
            
            # If the question is empty or very short, just get general advice
            if not question or len(question.strip()) < 3:
                return get_llm_advice(base_diagnosis)
                
            # Otherwise, include the question in the prompt
            return get_llm_advice(f"{base_diagnosis}. Question: {question}")
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I'm sorry, I encountered an error while generating a response. Please try again later."
    
    chat_btn.click(
        fn=respond_to_question,
        inputs=[chat_input, output_diagnosis],
        outputs=chat_output
    )

# Run the app
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, debug=True)
