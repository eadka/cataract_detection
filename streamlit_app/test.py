import streamlit as st

st.title("Uploader test")

uploaded_file = st.file_uploader(
    "Drag & drop an image here",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    st.success("File received!")
