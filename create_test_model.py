import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Create a simple CNN model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')  # 3 classes
    ])
    
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model

# Create and save the model
model = create_model()
model.save('sugercane1.keras')
print("Test model created and saved as 'sugercane1.keras'")

# Verify the model can be loaded
try:
    loaded_model = tf.keras.models.load_model('sugercane1.keras')
    print("Model loaded successfully!")
    loaded_model.summary()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print("\nIf you see errors, try installing the required packages:")
    print("pip install tensorflow-cpu==2.10.0 numpy<2.0.0")
