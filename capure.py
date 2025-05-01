import cv2
import numpy as np
import PIL.Image
import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import google.generativeai as genai
import os
from io import BytesIO
from dotenv import load_dotenv

# Initialize the Chroma client, image loader, and CLIP embedding function
dataset_folder = 'Data'
chroma_client = chromadb.PersistentClient(path="Vector_database")
image_loader = ImageLoader()
CLIP = OpenCLIPEmbeddingFunction()

# Create or get the image collection in the Chroma database
image_vdb = chroma_client.get_or_create_collection(name="image", embedding_function=CLIP, data_loader=image_loader)

# Setup the generative model for fashion suggestions
api_key = os.getenv("GOOGLE_API_KEY")  # Ensure you have set the environment variable GOOGLE_API_KEY
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("API key for Google Generative AI is not set. Please set the GOOGLE_API_KEY environment variable.")
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

# Load OpenCV's pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to capture image from webcam when a face is detected
def capture_image_from_webcam_when_person_appears():
    cap = cv2.VideoCapture(0)  # Access the webcam (0 is the default camera)
    st.write("Waiting for you to stand in front of the camera...")

    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:
            break

        # Convert the frame to grayscale (required for face detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # If faces are detected, capture the image
        if len(faces) > 0:
            # Draw rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Convert the frame to RGB (OpenCV uses BGR by default)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the webcam feed in Streamlit
            st.image(frame_rgb, channels="RGB", use_column_width=True)

            # Capture the image and exit the loop
            st.write("Person detected! Capturing image...")
            break

    cap.release()
    cv2.destroyAllWindows()

    # Return the captured image as a PIL Image
    return PIL.Image.fromarray(frame_rgb)

# Display a title and description in Streamlit
st.title("AI Fashion Styling Assistant with Webcam")
st.write("Stand in front of the webcam to get a fashion suggestion based on your appearance!")

# Capture button to initiate webcam feed
if st.button("Capture and Style"):
    # Capture the image once the person is detected
    captured_image = capture_image_from_webcam_when_person_appears()
    st.image(captured_image, caption="Captured Image", use_column_width=True)

    # Convert the captured image to a numpy array
    uploaded_image = np.array(captured_image)

    # Retrieve similar images from the Chroma database
    retrieved_imgs = image_vdb.query(query_images=[uploaded_image], include=['data'], n_results=3)

    # Display retrieved similar images
    st.subheader("Similar Images Retrieved from the Database:")
    if 'data' in retrieved_imgs and retrieved_imgs['data']:
        for i, img_data in enumerate(retrieved_imgs['data'][0]):
            try:
                # Convert numpy array to an image in the expected format
                img = PIL.Image.fromarray(img_data)
                st.image(img, caption=f"Similar Image {i+1}", use_column_width=True)
            except Exception as e:
                st.error(f"Error loading similar image {i+1}: {e}")
    else:
        st.write("No similar images found in the database.")

    # Generate styling recommendation based on the captured image
    prompt = ("You are a professional fashion and styling assistant. "
              "Analyze the provided image carefully and give detailed fashion advice, including how to style and complement this item. "
              "Offer suggestions for pairing it with accessories, footwear, and other clothing pieces. "
              "Based on the image, recommend how best to style this outfit to make a fashion statement.")

    try:
        response = model.generate_content([prompt, captured_image])
        st.subheader("Styling Recommendations:")
        st.write(response.text)
    except Exception as e:
        st.error(f"An error occurred while generating styling recommendations: {e}")








