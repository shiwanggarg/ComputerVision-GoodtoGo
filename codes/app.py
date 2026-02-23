import os
import json
import glob
import pickle
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

import insightface
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List

# =========================
# CONFIG
# =========================
BASE_DIR = "/Users/pranjalgarg/Desktop/test_images"
EXPORT_DIR = os.path.join(BASE_DIR, "export", "clusters")
DATA_DIR = os.path.join(BASE_DIR, "data")

FACEINFOS_PATH = os.path.join(DATA_DIR, "faceinfos.pkl")
IMAGE_INDEX_PATH = os.path.join(DATA_DIR, "image_index.json")
PERSON_INDEX_PATH = os.path.join(DATA_DIR, "person_index.json")
RECORDS_JSONL = os.path.join(DATA_DIR, "records.jsonl")

MODEL_ID = "gemini-2.5-flash"
SIM_THRESH = 0.67  # cosine similarity threshold for face match search

IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.webp", "*.bmp", "*.tiff", "*.gif")

# =========================
# HELPERS: Loading
# =========================
@st.cache_data
def load_face_records():
    with open(FACEINFOS_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

@st.cache_data
def load_records_jsonl(path):
    recs = []
    if not os.path.exists(path):
        return recs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except:
                pass
    return recs

def norm_text(s: str) -> str:
    s = s.lower().strip()
    s = " ".join(s.split())
    return s

# =========================
# HELPERS: Face matching
# =========================
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))

@st.cache_resource
def get_arcface_app():
    fa = insightface.app.FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])
    fa.prepare(ctx_id=0, det_size=(640, 640))
    return fa

def match_embedding(user_emb, face_records):
    user_emb = np.array(user_emb, dtype=np.float32)
    matches = []
    for r in face_records:
        emb = np.array(r["embedding"], dtype=np.float32)
        sim = cosine_sim(user_emb, emb)
        if sim >= SIM_THRESH:
            matches.append((sim, r))
    matches.sort(key=lambda x: x[0], reverse=True)
    return matches

# =========================
# HELPERS: Draw
# =========================
def draw_bboxes(img_path: str, boxes_with_labels):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for bbox, label in boxes_with_labels:
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], width=4)
        draw.text((x1, max(0, y1 - 15)), label)
    return img

# =========================
# Gemini: Query schema + parser
# =========================
class SearchQuery(BaseModel):
    people: List[int] = Field(default_factory=list, description="Person IDs (cluster labels). Example: [1,2]")
    same_image: bool = Field(default=False, description="If true, require all people appear in the same image.")
    tags: List[str] = Field(default_factory=list, description="Tags like 'black t-shirt', 'glasses', 'outdoor'.")
    top_k: int = Field(default=30, description="Max number of images to return (1..200).")

def gemini_parse_query(user_text: str) -> SearchQuery:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Run: export GEMINI_API_KEY='YOUR_KEY'")

    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=SearchQuery,
        temperature=0.0,
        max_output_tokens=256,
    )

    prompt = f"""
Convert the user's request into JSON following the schema.

Rules:
- "person 1", "person one" => people includes 1
- If user says: "together", "same photo", "same pic", "in same picture" => same_image=true
- Extract tags like: "black t-shirt", "glasses", "outdoor", "indoor"
- If user says "all photos of person X" => people=[X], same_image=false
- top_k default 30

User request:
{user_text}
"""

    resp = client.models.generate_content(model=MODEL_ID, contents=prompt, config=config)
    parsed = getattr(resp, "parsed", None)
    if parsed is None:
        return SearchQuery()

    parsed.people = [int(p) for p in parsed.people]
    parsed.tags = [norm_text(t) for t in parsed.tags if isinstance(t, str)]
    parsed.top_k = max(1, min(int(parsed.top_k), 200))
    return parsed

# =========================
# Retrieval (offline, no SQL)
# =========================
def build_maps(face_records):
    id_to_rec = {r["id"]: r for r in face_records}
    img_to_recs = {}
    for r in face_records:
        img_to_recs.setdefault(r["img_path"], []).append(r)
    return id_to_rec, img_to_recs

def search_local(query: SearchQuery, image_index, person_index, face_records, tagged_records):
    id_to_rec, img_to_recs = build_maps(face_records)

    candidate_images = None

    # --- People constraints ---
    if query.people:
        if query.same_image:
            wanted = set(query.people)
            imgs = []
            for img_path, meta in image_index.items():
                if wanted.issubset(set(meta.get("people", []))):
                    imgs.append(img_path)
            candidate_images = set(imgs)
        else:
            imgs = set()
            for p in query.people:
                for rid in person_index.get(str(p), []):
                    rec = id_to_rec.get(rid)
                    if rec:
                        imgs.add(rec["img_path"])
            candidate_images = imgs

    # --- Tag constraints ---
    if query.tags:
        wanted_tags = [norm_text(t) for t in query.tags]
        tag_imgs = set()

        for tr in tagged_records:
            img_path = tr.get("img_path")
            pid = tr.get("person_id")
            tags = [norm_text(t) for t in tr.get("tags", []) if isinstance(t, str)]

            if not img_path:
                continue

            # if user specified people, tag must match those people
            if query.people and pid not in query.people:
                continue

            if all(t in set(tags) for t in wanted_tags):
                tag_imgs.add(img_path)

        candidate_images = tag_imgs if candidate_images is None else (candidate_images & tag_imgs)

    if candidate_images is None:
        return [], "Give me at least a person id (person 1) or a tag (black t-shirt)."

    results = list(candidate_images)[: query.top_k]

    output = []
    for img_path in results:
        boxes = []
        if query.people:
            for r in img_to_recs.get(img_path, []):
                if r["cluster_label"] in query.people:
                    boxes.append((r["bbox"], f"person {r['cluster_label']}"))
        output.append((img_path, boxes))

    return output, None

# =========================
# UI
# =========================
st.set_page_config(page_title="CV Project: Face Clusters + Chat", layout="wide")
st.title("Face Cluster Explorer + Search + Chatbot")

# --- Check required files ---
missing = []
for p in [FACEINFOS_PATH, IMAGE_INDEX_PATH, PERSON_INDEX_PATH]:
    if not os.path.exists(p):
        missing.append(p)

if missing:
    st.error("Missing required files. Run cluster_faces.py first.\n\n" + "\n".join(missing))
    st.stop()

face_records = load_face_records()
image_index = load_json(IMAGE_INDEX_PATH)
person_index = load_json(PERSON_INDEX_PATH)
tagged_records = load_records_jsonl(RECORDS_JSONL)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Browse Clusters",
    "📷 Webcam Face Search",
    "⬆️ Upload Face Search",
    "💬 Chat Search"
])

# -------------------------
# TAB 1
# -------------------------
with tab1:
    st.header("Browse Cluster Export Folders")
    if not os.path.isdir(EXPORT_DIR):
        st.info("No export folder found yet. Run cluster_faces.py to create cluster exports.")
    else:
        cluster_folders = [d for d in sorted(os.listdir(EXPORT_DIR)) if os.path.isdir(os.path.join(EXPORT_DIR, d))]
        if not cluster_folders:
            st.info("No cluster folders found.")
        else:
            cluster = st.selectbox("Select cluster folder:", cluster_folders)
            selected_folder = os.path.join(EXPORT_DIR, cluster)

            image_files = []
            for ext in IMAGE_EXTS:
                image_files.extend(glob.glob(os.path.join(selected_folder, ext)))
            image_files = sorted(image_files)

            if not image_files:
                st.info("No images in this folder.")
            else:
                img_names = [os.path.basename(p) for p in image_files]
                img_name = st.selectbox("Select image:", img_names)
                img_path = os.path.join(selected_folder, img_name)
                st.image(Image.open(img_path), caption=f"{cluster} — {img_name}", use_container_width=True)

# -------------------------
# TAB 2
# -------------------------
with tab2:
    st.header("Webcam Face Search")
    arc_app = get_arcface_app()
    camera_img = st.camera_input("Scan your face")

    if camera_img is not None:
        img_array = np.asarray(Image.open(camera_img).convert("RGB"))
        faces = arc_app.get(img_array[:, :, ::-1])  # RGB->BGR
        if not faces:
            st.warning("No face detected. Try again.")
        else:
            user_emb = faces[0].embedding
            matches = match_embedding(user_emb, face_records)

            st.success(f"Found {len(matches)} matches (sim ≥ {SIM_THRESH}). Showing top 20.")
            for sim, r in matches[:20]:
                try:
                    annotated = draw_bboxes(
                        r["img_path"],
                        [(r["bbox"], f"person {r['cluster_label']} | sim {sim:.2f}")]
                    )
                    st.image(annotated, caption=os.path.basename(r["img_path"]), use_container_width=True)
                except Exception as e:
                    st.write(r["img_path"], e)

# -------------------------
# TAB 3
# -------------------------
with tab3:
    st.header("Upload Face Image to Search")
    arc_app = get_arcface_app()
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp", "gif", "tiff"])

    if uploaded_file is not None:
        img_array = np.asarray(Image.open(uploaded_file).convert("RGB"))
        faces = arc_app.get(img_array[:, :, ::-1])  # RGB->BGR
        if not faces:
            st.warning("No face detected in uploaded image.")
        else:
            user_emb = faces[0].embedding
            matches = match_embedding(user_emb, face_records)

            st.success(f"Found {len(matches)} matches (sim ≥ {SIM_THRESH}). Showing top 20.")
            for sim, r in matches[:20]:
                try:
                    annotated = draw_bboxes(
                        r["img_path"],
                        [(r["bbox"], f"person {r['cluster_label']} | sim {sim:.2f}")]
                    )
                    st.image(annotated, caption=os.path.basename(r["img_path"]), use_container_width=True)
                except Exception as e:
                    st.write(r["img_path"], e)

# -------------------------
# TAB 4
# -------------------------
with tab4:
    st.header("Chat Search (Gemini → Local Retrieval)")
    st.caption("Examples: 'person 1 and person 2 together' | 'person 3 black t-shirt' | 'all photos of person 5'")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        st.chat_message(msg["role"]).write(msg["content"])

    user_text = st.chat_input("Ask me to find photos…")
    if user_text:
        st.session_state.chat.append({"role": "user", "content": user_text})
        st.chat_message("user").write(user_text)

        try:
            q = gemini_parse_query(user_text)
        except Exception as e:
            st.session_state.chat.append({"role": "assistant", "content": f"Gemini parse error: {e}"})
            st.chat_message("assistant").write(f"Gemini parse error: {e}")
            st.stop()

        results, err = search_local(q, image_index, person_index, face_records, tagged_records)

        if err:
            st.session_state.chat.append({"role": "assistant", "content": err})
            st.chat_message("assistant").write(err)
        else:
            summary = f"Parsed as: people={q.people}, same_image={q.same_image}, tags={q.tags}. Found {len(results)} images."
            st.session_state.chat.append({"role": "assistant", "content": summary})
            st.chat_message("assistant").write(summary)

            for img_path, boxes in results:
                try:
                    if boxes:
                        annotated = draw_bboxes(img_path, boxes)
                        st.image(annotated, caption=os.path.basename(img_path), use_container_width=True)
                    else:
                        st.image(Image.open(img_path), caption=os.path.basename(img_path), use_container_width=True)
                except Exception as e:
                    st.write(img_path, e)
