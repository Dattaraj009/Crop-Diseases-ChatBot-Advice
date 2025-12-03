import gradio as gr
import numpy as np
from PIL import Image
import os
import json
import re
import requests
import base64
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from io import BytesIO

# Class names
CLASS_NAMES = ["Normal leaf", "Red rot", "White leaf"]

# Load knowledge base
with open('knowledge_base.json', 'r') as f:
    KNOWLEDGE_BASE = json.load(f)

def get_disease_info(disease_name):
    """Get information about a disease from the knowledge base"""
    if not disease_name:
        return None
        
    disease_name_lower = disease_name.lower()
    for disease in KNOWLEDGE_BASE['diseases']:
        # Check if the disease name or scientific name contains the search term
        if (disease_name_lower in disease['name'].lower() or 
            disease_name_lower in disease.get('scientific_name', '').lower() or
            any(disease_name_lower in s.lower() for s in disease.get('symptoms', [])) or
            any(disease_name_lower in c.lower() for c in disease.get('causes', []))):
            return disease
    return None

# Image preprocessing
def preprocess_image(img):
    """Preprocess the image for the model"""
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def load_model():
    """Load the trained model"""
    model_path = "sugercane2.keras"
    
    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
        raise FileNotFoundError(f"Model file {model_path} not found or is empty. Please ensure the model file exists.")
    
    try:
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure the model file is not corrupted")
        print("2. Try re-saving the model with: model.save('sugercane2.keras')")
        print("3. Check TensorFlow version compatibility")
        raise

# Load the model when the app starts
print("Starting application...")
model = load_model()

def predict_disease(img):
    """Predict disease using the trained model"""
    try:
        # Preprocess the image
        processed_img = preprocess_image(img)
        
        # Get predictions
        predictions = model.predict(processed_img, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Get class name
        class_name = CLASS_NAMES[predicted_class_idx]
            
        return class_name, confidence
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return "Prediction Error", 0.0

# Function to get advice using Ollama
def get_llm_advice(diagnosis):
    """Get advice from LLM based on the diagnosis"""
    try:
        # Prepare the prompt based on whether it's a normal leaf or disease
        if "normal" in diagnosis.lower():
            prompt = """You are a sugarcane cultivation expert. Provide a SINGLE, COMPLETE response for a healthy sugarcane plant. 
            
            RULES:
            - DO NOT ask any questions
            - DO NOT request additional information
            - Provide ALL necessary information in this response
            - Keep it under 250 words
            - Use clear section headers
            - Be specific to sugarcane
            
            FORMAT:
            ✅ **Healthy Sugarcane Plant**
            
            ⚠️ **Watch For**
            - [key warning signs]
            
            Remember: This is a COMPLETE response. Do not ask for more information.
            """
        else:
            prompt = f"""You are an expert in plant pathology. Provide specific care advice for a plant with the following condition: {diagnosis}.
            
            Your response should be factual and practical, including:
            1. A brief description of the condition
            2. Recommended treatment steps
            3. Preventive measures
            4. When to consult a professional
            
            Be concise but thorough in your advice. If you're not certain about the diagnosis, say so."""

        
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
    """Analyze image and return diagnosis and advice with confidence score"""
    if model is None:
        return "Error: Model not loaded. Please check the logs.", ""
    
    try:
        # Get prediction from the model
        prediction, confidence = predict_disease(img)
        
        # Get advice from LLM
        advice = get_llm_advice(prediction)
        
        # Return prediction with confidence score as percentage
        return f"{prediction} (Confidence: {confidence*100:.1f}%)", advice
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        return f"Error: {str(e)}", ""

# Create the Gradio interface
with gr.Blocks(title="Crop Health Advisor") as demo:
    gr.Markdown("""
    # 🌱 Crop Health Advisor
    Upload an image of your crop to get a diagnosis and care advice.
    """)
    
    # Custom CSS to ensure labels are visible and make follow-up section resizable
    custom_css = """
    /* Style for labels */
    .gr-form {
        margin-bottom: 12px !important;
    }
    
    /* Make labels visible */
    .gr-box label {
        display: block !important;
        font-weight: 600 !important;
        color: #1e293b !important;
        margin-bottom: 4px !important;
    }
    
    /* Default style for text areas (not resizable) */
    .gradio-textbox textarea {
        resize: none !important;
        min-height: 60px !important;
        width: 100% !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 4px !important;
        padding: 8px !important;
    }
    
    /* Make only the follow-up section's response box resizable */
    #chat_output textarea {
        resize: vertical !important;
        min-height: 100px !important;
        max-height: 500px !important;
        overflow-y: auto !important;
    }
    """
    demo.css = custom_css
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Crop Image")
            submit_btn = gr.Button("Analyze")
        
        with gr.Column():
            output_diagnosis = gr.Textbox(label="Diagnosis", interactive=False, show_label=True, show_copy_button=True, container=True)
            output_advice = gr.Textbox(label="Care Advice", lines=4, interactive=False, show_label=True, show_copy_button=True, container=True)
    
    # Chat interface for follow-up questions
    with gr.Accordion("Ask a follow-up question", open=False):
        chat_input = gr.Textbox(label="Your question", placeholder="Ask about treatment options...")
        chat_output = gr.Textbox(label="Response", interactive=False, elem_id="chat_output")
        chat_btn = gr.Button("Ask")
    
    # Define button actions
    submit_btn.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=[output_diagnosis, output_advice]
    )
    
    def respond_to_question(question, diagnosis):
        """Generate a response to the user's question using the knowledge base and LLM"""
        if not diagnosis:
            return "Please analyze an image first so I can provide relevant advice."
        
        try:
            # Get the base diagnosis without the confidence score
            base_diagnosis = diagnosis.split('(')[0].strip()
            
            # If the question is empty or very short, ask for more details
            if not question or len(question.strip()) < 3:
                return "Please ask a specific question about the diagnosis or treatment options."
            
            # Get disease information from knowledge base
            disease_info = get_disease_info(base_diagnosis)
            
            # Common question patterns and their corresponding responses
            question_lower = question.lower()
            
            if disease_info:
                # Handle specific question types using knowledge base
                if any(q in question_lower for q in ['cure', 'treat', 'treatment', 'solution']):
                    if 'treatment' in disease_info:
                        return "\n".join(["✅ Treatment options:"] + [f"• {t}" for t in disease_info['treatment']])
                
                elif any(q in question_lower for q in ['prevent', 'prevention', 'avoid']):
                    if 'prevention' in disease_info:
                        return "\n".join(["🛡️ Prevention measures:"] + [f"• {p}" for p in disease_info['prevention']])
                
                elif any(q in question_lower for q in ['symptom', 'sign', 'look like']):
                    if 'symptoms' in disease_info:
                        return "\n".join(["⚠️ Common symptoms:"] + [f"• {s}" for s in disease_info['symptoms']])
                
                elif any(q in question_lower for q in ['cause', 'reason', 'why']):
                    if 'causes' in disease_info:
                        return "\n".join(["🔍 Possible causes:"] + [f"• {c}" for c in disease_info['causes']])
                
                elif 'curable' in question_lower:
                    status = "Yes" if disease_info.get('is_curable', False) else "No"
                    return f"Curable: {status}. " + ("Early treatment improves success rates." if status == "Yes" else "Focus on prevention and management.")
            
            # If no specific pattern matched or disease not found, use LLM with context
            context = json.dumps(disease_info, indent=2) if disease_info else "No specific information available"
            
            prompt = f"""You are a sugarcane disease expert. Use the following information to answer the question.
            
            Disease: {base_diagnosis}
            Context: {context}
            
            Question: {question}
            
            Guidelines:
            - Be specific and concise (under 150 words)
            - Only use information from the provided context
            - If the question can't be answered from context, say so
            - Do not make up information
            - Format lists with bullet points
            - Do not ask follow-up questions
            """
            
            # Make the API call to Ollama
            data = {
                "model": "gemma3",
                "prompt": prompt,
                "stream": False,
                "max_tokens": 300
            }
            
            response = requests.post("http://localhost:11434/api/generate", json=data)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "I couldn't generate a response. Please try again.")
            
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
