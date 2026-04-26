# import tflite_runtime.interpreter as tflite
from ai_edge_litert.interpreter import Interpreter
import streamlit as st
import numpy as np
from PIL import Image

# Configuration
MODEL_PATH = "leaf_lens_model.tflite"
# Update this list to match the labels your model was trained on
LABELS = ["Tomato___Bacterial_spot", "Tomato___Septoria_leaf_spot", "Tomato___Late_blight", "Corn_(maize)___Common_rust_", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"] 

@st.cache_resource
def load_model(path):
    interpreter = Interpreter(model_path=path)
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
st.set_page_config(page_title="Leaf Len", page_icon="🌿")
st.title("🌿 Leaf Len - Plant Disease Detector")
st.write("Upload an image of a tomato or corn leaf to analyze its health.")
st.write("1. Tomato Bacterial Spot: Caused by Xanthomonas bacteria, this disease creates small, dark, water-soaked spots on leaves and fruit. It thrives in warm, wet conditions and can lead to significant defoliation, exposing fruit to sunscald.")
st.write("2. Tomato Septoria Leaf Spot: A fungal infection that typically starts on the lower leaves of the plant. It is identified by small, circular spots with dark borders and tan or gray centers, often containing tiny black specks (fruiting bodies).")
st.write("3. Tomato Late Blight: A highly destructive water mold disease (Phytophthora infestans) that spreads rapidly in cool, damp weather. It causes large, dark, greasy-looking patches on leaves and can rot entire fruits and stems in just a few days.")
st.write("4. Corn Common Rust: Identified by small, cinnamon-brown pustules that erupt on both the upper and lower surfaces of corn leaves. These pustules contain powdery spores that are easily spread by wind to neighboring plants.")
st.write("5. Corn Gray Leaf Spot: Caused by the fungus Cercospora zeae-maydis, this disease produces long, narrow, rectangular lesions that run parallel to the leaf veins. As they mature, the spots turn gray and can merge, killing large areas of leaf tissue.")
st.write("⚠️ Note: If you upload a picture of a corn or tomato leaf and it show for example, Class 7 instead of a type of disease, it means that it is the class of a disease that the model hasn't trained on.")

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