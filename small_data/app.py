import os
import streamlit as st
import numpy as np
from PIL import Image
import pickle
import insightface
import glob
st.set_page_config(layout="wide")

EXPORT_DIR = '/Users/pranjalgarg/Desktop/test_image_recogo/arcface_clusters_export'

# Load all face info for search
with open("faceinfos.pkl", "rb") as f:
    face_infos = pickle.load(f)  # List of dicts: img_path, embedding, cluster_label, bbox

# Initialize ArcFace for embedding extraction
app = insightface.app.FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

st.title("Face Cluster Explorer & Search")

tab1, tab2, tab3 = st.tabs([
    "🔍 Browse All Clusters",
    "📷 Webcam Face Search",
    "⬆️ Upload Image and Search"
])

# Extensions to use in cluster search
IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png")

with tab1:
    st.header("Browse Clusters")
    # Show only cluster folders that contain at least one image of supported type
    cluster_folders = [
        d for d in sorted(os.listdir(EXPORT_DIR))
        if os.path.isdir(os.path.join(EXPORT_DIR, d)) and (
            sum([len(glob.glob(os.path.join(EXPORT_DIR, d, ext))) for ext in IMAGE_EXTS]) > 0
        )
    ]

    if cluster_folders:
        cluster = st.selectbox("Select cluster folder:", cluster_folders)
        selected_folder = os.path.join(EXPORT_DIR, cluster)
        image_files = []
        for ext in IMAGE_EXTS:
            image_files.extend(glob.glob(os.path.join(selected_folder, ext)))
        image_files = sorted(image_files)
        img_names = [os.path.basename(p) for p in image_files]
        img = st.selectbox("Select image to view:", img_names)

        if img and img_names:
            img_path = os.path.join(selected_folder, img)
            try:
                st.image(Image.open(img_path),
                         caption=f'Cluster: {cluster}, File: {img}',
                         use_container_width=True)
            except Exception as e:
                st.error(f"Could not open image: {img_path}. Error: {e}")
    else:
        st.info("No non-empty clusters found.")

with tab2:
    st.header("Scan Face with Webcam")
    camera_img = st.camera_input("Scan your face (Webcam)")
    if camera_img is not None:
        try:
            img_array = np.asarray(Image.open(camera_img))
            faces = app.get(img_array)
            if faces:
                user_emb = faces[0].embedding
                threshold = 0.33
                results = []
                for info in face_infos:
                    dist = np.dot(user_emb, info['embedding']) / (
                        np.linalg.norm(user_emb) * np.linalg.norm(info['embedding']))
                    if dist > (1 - threshold):
                        results.append(info)
                st.success(f"Found {len(results)} matches in clusters!")
                for r in results:
                    try:
                        st.image(
                            r['img_path'],
                            caption=f"Cluster: {r['cluster_label']}, File: {os.path.basename(r['img_path'])}",
                            use_container_width=True
                        )
                        st.write(f"BBox: {r['bbox']}")
                    except Exception as e:
                        st.error(f"Could not open image: {r['img_path']}. Error: {e}")
            else:
                st.warning("No face detected. Try again!")
        except Exception as e:
            st.error(f"Could not process webcam image: {e}")

with tab3:
    st.header("Upload Face Image to Search")
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp", "bmp", "gif", "tiff"]
    )
    if uploaded_file is not None:
        try:
            img_array = np.asarray(Image.open(uploaded_file))
            faces = app.get(img_array)
            if faces:
                user_emb = faces[0].embedding
                threshold = 0.33
                found_clusters = set()
                results = []
                for info in face_infos:
                    dist = np.dot(user_emb, info['embedding']) / (
                        np.linalg.norm(user_emb) * np.linalg.norm(info['embedding']))
                    if dist > (1 - threshold):
                        results.append(info)
                        found_clusters.add(info['cluster_label'])
                st.success(f"Found {len(results)} matches in clusters: {sorted(found_clusters)}")
                for r in results:
                    try:
                        st.image(
                            r['img_path'],
                            caption=f"Cluster: {r['cluster_label']}, File: {os.path.basename(r['img_path'])}",
                            use_container_width=True
                        )
                        st.write(f"BBox: {r['bbox']}")
                    except Exception as e:
                        st.error(f"Could not open image: {r['img_path']}. Error: {e}")
            else:
                st.warning("No face detected in uploaded image.")
        except Exception as e:
            st.error(f"Could not process uploaded image: {e}")

#   streamlit run app.py

