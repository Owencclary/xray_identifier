import streamlit as st

from PIL import Image
from model import shapley, preprocess_single_image, predict_single_image, model, fracture_dict, class_names_dict

# Custom CSS to set the background image and styles
st.markdown(
    """ <style>
        .title {
            font-size: 36px;
            color: #2E86C1;
            font-weight: bold;
            text-align: center;
        }
        .text-box {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            margin: 20px;
            text-align: center;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .fracture-description-box {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            margin: 20px;
        }
        .stApp {
            background-image: url("https://static.vecteezy.com/system/resources/previews/021/430/771/non_2x/clean-sky-blue-gradient-background-with-text-space-editable-blurred-white-blue-illustration-for-the-backdrop-of-the-banner-poster-business-presentation-book-cover-advertisement-or-website-vector.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }
       </style> """,
    unsafe_allow_html=True
)

# Title and file input
st.sidebar.markdown('<div class="title">X-Ray Fracture Identifier</div>', unsafe_allow_html=True)
st.sidebar.write('') # Spacing
st.sidebar.write('') # Spacing

# Display file uploader and save uploaded file to variable
st.sidebar.write('Step 1: Upload X-ray for analysis')
uploaded_file = st.sidebar.file_uploader("", type=["jpg", "jpeg", "png"])
st.sidebar.write('') # Spacing
st.sidebar.write('') # Spacing


# Create columns
col1, col2 = st.columns(2)

# Add titles to columns
col1.markdown('<div class="text-box"><h2>X-Ray</h2></div>', unsafe_allow_html=True)
col2.markdown('<div class="text-box"><h2>Diagnosis</h2></div>', unsafe_allow_html=True)

# Text to guide user
st.sidebar.write('Step 2: Click here to run analysis')
st.sidebar.write('') # Spacing

# Button to analyze fracture
if st.sidebar.button("Analyze Fracture") and uploaded_file is not None:

    # Display fracture
    col1.image(uploaded_file, caption='Uploaded X-Ray', use_column_width=True)

    # Turn file into image, preprocess it
    img = Image.open(uploaded_file)
    img_array = preprocess_single_image(img)
    print("Image preprocessed successfully.")

    # Predict image
    predicted_fracture = predict_single_image(model, img_array, class_names_dict)
    print(f"Predicted class: {predicted_fracture}")

    # Display fracture prediction
    if predicted_fracture == 'non_fractured':
        col2.markdown(f'<div class="text-box"><h2>No Fracture Detected</h2</div>', unsafe_allow_html=True)
    else:
        # Display fracture with description
        fracture_description = fracture_dict[predicted_fracture]  # Get description from fracture_descriptions.py file
        col2.markdown(f'<div class="text-box"><h2>{predicted_fracture}</h2></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="fracture-description-box"><h4>Fracture Description: </h4>{fracture_description}</div>', unsafe_allow_html=True)

        shapley(img_array, col2) # display shapley
