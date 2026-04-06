import numpy as np
from utils import *

def load_weights(filename="load_weight.py"):
    """Load weights from a .npz file saved from any NumPy MNIST training repo"""
    data = np.load(filename)
    W1 = data['W1']      # shape: (784, 128)
    b1 = data['b1']      # shape: (128,)
    W2 = data['W2']      # shape: (128, 10)
    b2 = data['b2']      # shape: (10,)
    print("✅ Weights loaded successfully!")
    return W1, b1, W2, b2

# ====================== FORWARD PROPAGATION ======================
def forward_propagation(X, W1, b1, W2, b2):
    """
    X: input image (shape: (1, 784) or (n_samples, 784), normalized 0-1)
    Returns: predicted probabilities for each digit 0-9
    """
    # Hidden layer
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    # Output layer
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return A2

# ====================== PREDICT DIGIT ======================
def predict_digit(image, W1, b1, W2, b2):
    """
    image: 28x28 numpy array (or flattened 784)
    Returns: predicted digit + confidence
    """
    # Preprocess: flatten and normalize
    if image.ndim == 2:  # 28x28 image
        X = image.reshape(1, 784) / 255.0
    else:
        X = image.reshape(1, -1) / 255.0
    
    probabilities = forward_propagation(X, W1, b1, W2, b2)
    predicted_digit = np.argmax(probabilities)
    confidence = np.max(probabilities) * 100
    
    return predicted_digit, confidence, probabilities

# ====================== EXAMPLE USAGE ======================
if __name__ == "__main__":
    # 1. Load pre-trained weights
    W1, b1, W2, b2 = load_weights("load_weights.py")
    
    # 2. Example: Use your own handwritten digit (replace with real image)
    # For testing, you can create a dummy image or load from MNIST
    print("\n🔍 Testing with a random dummy image...")
    dummy_image = np.random.rand(28, 28) * 0.3   # fake image (replace with real!)
    
    digit, confidence, probs = predict_digit(dummy_image, W1, b1, W2, b2)
    
    print(f"Predicted Digit: {digit}")
    print(f"Confidence: {confidence:.2f}%")
    print("Probabilities:", np.round(probs[0], 4))
