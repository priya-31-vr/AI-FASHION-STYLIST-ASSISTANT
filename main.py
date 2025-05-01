import streamlit as st
import numpy as np
import PIL.Image
import io
import requests
import google.generativeai as genai
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import chromadb
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings("ignore")

# Load environment variables and configure Gemini
load_dotenv()
api_key = os.getenv("api_key")
genai.configure(api_key=api_key)

# Helper to format image paths from query results
def format_image_inputs(data):
    image_path_1 = data['uris'][0][0]
    image_path_2 = data['uris'][0][1]
    return [image_path_1, image_path_2]

# Helper to open images from path or bytes
def open_image(img_data):
    if isinstance(img_data, str):
        response = requests.get(img_data)
        img = PIL.Image.open(io.BytesIO(response.content))
    elif isinstance(img_data, np.ndarray):
        img = PIL.Image.fromarray(img_data.astype('uint8'))
    elif isinstance(img_data, list):
        img_data = np.array(img_data, dtype='uint8')
        img = PIL.Image.fromarray(img_data)
    else:
        raise ValueError("Unsupported image data format")
    return img

# Streamlit interface
st.set_page_config(page_title="AI Fashion Stylist", layout="centered")
st.title("👗 AI Fashion Styling Assistant")
st.write("Upload an image or describe your look to get fashion advice & style recommendations.")

uploaded_file = st.file_uploader("📸 Upload an image:", type=["jpg", "jpeg", "png"])
query = st.text_input("📝 Or enter your styling query:")

if st.button("✨ Get Styling Recommendations"):
    with st.spinner("Analyzing fashion and preparing your recommendations..."):
        # Connect to ChromaDB
        chroma_client = chromadb.PersistentClient(path="Vector_database")
        image_loader = ImageLoader()
        CLIP = OpenCLIPEmbeddingFunction()
        image_vdb = chroma_client.get_or_create_collection(
            name="image",
            embedding_function=CLIP,
            data_loader=image_loader
        )

        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        styling_prompt = (
            "You are a professional fashion and styling assistant with expertise in personalized outfit advice. "
            "Analyze the provided image carefully and give detailed styling tips including accessory ideas, matching clothes, and suggestions."
        )

        # Handle uploaded image
        if uploaded_file is not None:
            uploaded_image = np.array(PIL.Image.open(uploaded_file))

            try:
                retrieved_imgs = image_vdb.query(
                    query_images=[uploaded_image],
                    include=['data'],
                    n_results=3
                )

                st.subheader("🖼️ Similar Looks Retrieved:")
                for i, img_data in enumerate(retrieved_imgs['data'][0]):
                    img = open_image(img_data)
                    st.image(img, caption=f"Similar Style {i+1}", use_container_width=True)

                    # Generate fashion advice using Gemini
                    try:
                        response = model.generate_content([styling_prompt, img])
                        st.subheader("👠 Styling Tips:")
                        st.write(response.text)
                    except Exception as e:
                        st.error(f"❌ Error with Gemini: {e}")
            except Exception as e:
                st.error(f"❌ Failed to retrieve similar images: {e}")

        # Handle text-based query
        elif query:
            try:
                results = image_vdb.query(
                    query_texts=[query],
                    n_results=2,
                    include=['uris', 'distances']
                )
                image_paths = format_image_inputs(results)

                img1 = PIL.Image.open(image_paths[0])
                img2 = PIL.Image.open(image_paths[1])

                st.image(img1, caption="Match 1", use_container_width=True)
                st.image(img2, caption="Match 2", use_container_width=True)

                text_query_prompt = (
                    styling_prompt + f" This is the user's styling query: {query}. "
                    "Based on the outfit vibes of the matched images, give fashion suggestions."
                )

                try:
                    response = model.generate_content([text_query_prompt, img1, img2])
                    st.subheader("🧥 Outfit Recommendations:")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"❌ Error generating text-based suggestions: {e}")

            except Exception as e:
                st.error(f"❌ Query error: {e}")
        else:
            st.warning("⚠️ Please upload an image or enter a styling query.")
