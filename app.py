import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time

# Page configuration
st.set_page_config(
    page_title="EcoSort-AI",
    page_icon="♻️",
    layout="centered"
)

# Custom styling
st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.stButton button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 16px;
}

.prediction-box {
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    color: white;
}

.footer {
    text-align: center;
    margin-top: 50px;
    color: gray;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# Load trained model
model = tf.keras.models.load_model(
    "models/mobilenetv2_model.keras"
)

# Class names
class_names = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation"
]

# Class colors
class_colors = {
    "Cardboard": "#A68A64",
    "Food Organics": "#2D6A4F",
    "Glass": "#3A86FF",
    "Metal": "#6C757D",
    "Miscellaneous Trash": "#9D4EDD",
    "Paper": "#F4E285",
    "Plastic": "#FFBE0B",
    "Textile Trash": "#E76F51",
    "Vegetation": "#40916C"
}

# Sidebar
st.sidebar.title("♻️ EcoSort-AI")

st.sidebar.markdown("""
### About
EcoSort-AI is an intelligent waste classification system powered by TensorFlow and MobileNetV2.

### Technologies Used
- TensorFlow
- MobileNetV2
- Streamlit
- Python
- Deep Learning

### Model Features
✅ Waste Image Classification  
✅ Transfer Learning  
✅ Confidence Scores  
✅ Top-3 Predictions  
✅ Interactive Web Interface  
""")

# Main title
st.title("♻️ EcoSort-AI")

st.markdown("""
Upload a waste image and let the AI classify it instantly using deep learning.
""")

# Upload image
uploaded_file = st.file_uploader(
    "📤 Drag and drop an image here",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    img = Image.open(uploaded_file)
    # convert to RGB 
    img = img.convert("RGB")

    # Display image
    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Predict button
    if st.button("🔍 Predict Waste Type"):

        # Loading spinner
        with st.spinner("Analyzing image..."):

            time.sleep(1)

            # Resize image
            resized_img = img.resize((128, 128))

            # Convert image to array
            img_array = image.img_to_array(resized_img)

            # Expand dimensions
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            prediction = model.predict(img_array)

            # Get top 3 predictions
            top_3_indices = prediction[0].argsort()[-3:][::-1]

            top_3_classes = [
                class_names[i]
                for i in top_3_indices
            ]

            top_3_confidences = [
                prediction[0][i] * 100
                for i in top_3_indices
            ]

            predicted_class = top_3_classes[0]
            confidence = top_3_confidences[0]

            # Get color for predicted class
            box_color = class_colors.get(
                predicted_class,
                "#1B4332"
            )

        # Success message
        st.success("Prediction completed successfully!")

        # Main prediction
        st.subheader(" Main Prediction")

        st.markdown(
    f"""
    <div style="
        background-color:{box_color};
        padding:20px;
        border-radius:15px;
        color:white;
        margin-top:20px;
    ">

    <div style="
        font-size:32px;
        font-weight:bold;
    ">
        {predicted_class}
    </div>

    <div style="
        font-size:18px;
        margin-top:10px;
    ">
        {confidence:.2f}% confidence
    </div>

    </div>
    """,
    unsafe_allow_html=True
)

        # Confidence bar
        st.subheader("📊 Confidence Level")

        st.progress(int(confidence))

        # Confidence feedback
        if confidence >= 80:
            st.success("High confidence prediction")
        elif confidence >= 60:
            st.warning("Moderate confidence prediction")
        else:
            st.error("Low confidence — result may be uncertain")

        # Top 3 predictions
        st.subheader("🏆 Top 3 Predictions")

        for i in range(3):

            st.write(
                f"{i+1}. {top_3_classes[i]} — "
                f"{top_3_confidences[i]:.2f}%"
            )

# Footer
st.markdown("""
<div class="footer">
Developed by Salam Atallah |
© 2024 EcoSort-AI
</div>
""", unsafe_allow_html=True)
