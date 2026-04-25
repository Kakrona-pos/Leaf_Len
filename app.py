import tensorflow as tf
import numpy as np
from PIL import Image

# Configuration
MODEL_PATH = "leaf_lens_model.tflite"
# Update this list to match the labels your model was trained on
LABELS = ["Healthy", "Early Blight", "Late Blight", "Leaf Spot"] 

@st.cache_resource
def load_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

def predict(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get expected input shape (e.g., 224x224)
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # Preprocessing
    img = image.resize((width, height))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize if necessary (standard for many TFLite models)
    img_array = img_array / 255.0

    # Inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# --- UI Interface ---
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")
st.title("🌿 Plant Disease Detection")
st.write("Upload an image of a leaf to analyze its health using the LeafLens model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Leaf Image', use_container_width=True)
    
    if st.button("Run Detection"):
        with st.spinner("Analyzing..."):
            try:
                interpreter = load_model(MODEL_PATH)
                results = predict(interpreter, image)
                
                # Process results
                top_index = np.argmax(results[0])
                confidence = results[0][top_index]
                
                st.subheader("Result:")
                if confidence > 0.5:
                    label = LABELS[top_index] if top_index < len(LABELS) else f"Class {top_index}"
                    st.success(f"**{label}** ({confidence*100:.2f}%)")
                else:
                    st.warning("Inconclusive result. Please try a clearer photo.")
                    
            except Exception as e:
                st.error(f"Error during inference: {e}")