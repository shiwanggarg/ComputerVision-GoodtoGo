import os
import cv2
import json
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pillow_heif
pillow_heif.register_heif_opener()

from PIL import Image
from sklearn.cluster import DBSCAN
from insightface.app import FaceAnalysis

# =========================
# CONFIG
# =========================
BASE_DIR = "/Users/pranjalgarg/Desktop/test_images"
IMAGE_FOLDER = os.path.join(BASE_DIR, "small_dataset")
EXPORT_DIR = os.path.join(BASE_DIR, "export", "clusters")
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

SUPPORTED_EXTS = ('.jpg', '.jpeg', '.png', '.heic', '.bmp', '.tiff', '.webp', '.gif')

# DBSCAN clustering params
EPS = 0.35
MIN_SAMPLES = 2

# InsightFace model
app = FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

color_map = [
    (0,0,255), (0,255,0), (255,0,0), (0,165,255), (128,0,128), (255,255,0),
    (255,0,255), (203,192,255), (0,255,255), (50,205,50), (255,255,240), (128,128,128)
]

def get_color(idx: int):
    return color_map[idx % len(color_map)] if idx != -1 else (0,0,0)

def load_images(folder: str):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"IMAGE_FOLDER not found: {folder}")
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(SUPPORTED_EXTS)
    ]

def pil_to_bgr(img_path: str) -> np.ndarray:
    img_pil = Image.open(img_path)
    img = np.array(img_pil)

    # grayscale
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # RGBA
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # PIL RGB -> OpenCV BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    n = np.linalg.norm(v) + 1e-12
    return v / n

def main():
    img_paths = load_images(IMAGE_FOLDER)
    print(f"Loaded {len(img_paths)} images from: {IMAGE_FOLDER}")

    embeddings = []
    face_infos_raw = []  # temporary list with bbox + embedding + path

    # --------- Face detection + embeddings ----------
    for img_path in tqdm(img_paths, desc="Extracting embeddings"):
        try:
            img_bgr = pil_to_bgr(img_path)
        except Exception as e:
            print(f"Could not load {img_path}: {e}")
            continue

        faces = app.get(img_bgr)
        if not faces:
            continue

        for face in faces:
            emb = normalize(face.embedding)
            bbox = face.bbox.astype(int)
            embeddings.append(emb)
            face_infos_raw.append({
                "img_path": img_path,
                "bbox": bbox,
                "embedding": emb
            })

    if len(embeddings) == 0:
        print("No faces found in dataset. Put images into images/ and rerun.")
        return

    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"Extracted {len(embeddings)} face embeddings.")

    # --------- Clustering ----------
    db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric='cosine')
    labels = db.fit_predict(embeddings)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Clusters found: {n_clusters}, Noise faces: {(labels == -1).sum()}")

    # --------- Build final records with stable IDs ----------
    # record ID = filename + face index (unique enough)
    face_records = []
    image_index = defaultdict(lambda: {"people": set(), "records": []})
    person_index = defaultdict(list)

    for i, (info, label) in enumerate(zip(face_infos_raw, labels)):
        rid = f"{os.path.basename(info['img_path'])}__face{i}"
        rec = {
            "id": rid,
            "img_path": info["img_path"],
            "cluster_label": int(label),
            "bbox": info["bbox"].tolist(),
            "embedding": info["embedding"].tolist(),
        }
        face_records.append(rec)

        image_index[info["img_path"]]["people"].add(int(label))
        image_index[info["img_path"]]["records"].append(rid)
        person_index[str(int(label))].append(rid)

    # --------- Export marked images per cluster ----------
    cluster_map = defaultdict(list)
    for rec in face_records:
        cluster_map[rec["cluster_label"]].append(rec)

    for clus_id, recs in cluster_map.items():
        clus_name = f"cluster_{clus_id}" if clus_id != -1 else "noise"
        clus_folder = os.path.join(EXPORT_DIR, clus_name)
        os.makedirs(clus_folder, exist_ok=True)

        # group bboxes by image
        img_face_map = defaultdict(list)
        for r in recs:
            img_face_map[r["img_path"]].append(r["bbox"])

        for img_path, bboxes in img_face_map.items():
            try:
                img_bgr = pil_to_bgr(img_path)
            except Exception as e:
                print(f"Could not load {img_path} for marking: {e}")
                continue

            color = get_color(clus_id)
            for box in bboxes:
                x1, y1, x2, y2 = box
                label_txt = f"Cluster {clus_id}" if clus_id != -1 else "Noise"
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)
                cv2.putText(
                    img_bgr, label_txt, (x1, max(10, y1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA
                )

            out_base = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(clus_folder, out_base + ".png")
            cv2.imwrite(out_path, img_bgr)

    # --------- Save offline data ----------
    with open(os.path.join(DATA_DIR, "faceinfos.pkl"), "wb") as f:
        pickle.dump(face_records, f)

    with open(os.path.join(DATA_DIR, "image_index.json"), "w") as f:
        json.dump(
            {k: {"people": sorted(list(v["people"])), "records": v["records"]} for k, v in image_index.items()},
            f, indent=2
        )

    with open(os.path.join(DATA_DIR, "person_index.json"), "w") as f:
        json.dump(person_index, f, indent=2)

    print("\nSaved:")
    print(f" - {os.path.join(DATA_DIR, 'faceinfos.pkl')}")
    print(f" - {os.path.join(DATA_DIR, 'image_index.json')}")
    print(f" - {os.path.join(DATA_DIR, 'person_index.json')}")
    print(f"Exported marked cluster images to: {EXPORT_DIR}")

if __name__ == "__main__":
    main()
