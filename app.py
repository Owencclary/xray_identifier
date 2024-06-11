import streamlit as st
from model import analyze_fracture

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
st.sidebar.write('')  # Spacing

uploaded_file = st.sidebar.file_uploader("Input X-Ray Image for Processing: ", type=["jpg", "jpeg", "png"])

# Move the button to the sidebar
if st.sidebar.button("Analyze Fracture") and uploaded_file is not None:
    st.write("Button clicked, analyzing fracture...")
    diagnosis_result = analyze_fracture(uploaded_file)
    st.write("Analysis complete.")

# Create columns with a gap for the vertical line
col1, col2 = st.columns(2)

# Add content to the columns with a white background
with col1:
    st.markdown('<div class="text-box"><h2>X-Ray</h2></div>', unsafe_allow_html=True)
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded X-Ray', use_column_width=True)

col2.markdown('<div class="text-box"><h2>Diagnosis</h2></div>', unsafe_allow_html=True)
