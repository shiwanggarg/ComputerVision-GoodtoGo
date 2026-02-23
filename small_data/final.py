import os
import cv2
import numpy as np
import pillow_heif                     # HEIC/HEIF support
pillow_heif.register_heif_opener()     # Enable HEIC for PIL
from PIL import Image
from sklearn.cluster import DBSCAN
import pickle
import insightface
from insightface.app import FaceAnalysis


SUPPORTED_EXTS = ('.jpg', '.jpeg', '.png', '.heic', '.bmp', '.tiff', '.webp', '.gif')

# Initialize ArcFace model/app
app = FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

IMAGE_FOLDER = '/Users/pranjalgarg/Desktop/test_image_recogo/dataset'
EXPORT_DIR = '/Users/pranjalgarg/Desktop/test_image_recogo/arcface_clusters_export'
os.makedirs(EXPORT_DIR, exist_ok=True)

color_map = [
    (0,0,255), (0,255,0), (255,0,0), (0,165,255), (128,0,128), (255,255,0),
    (255,0,255), (203,192,255), (0,255,255), (50,205,50), (255,255,240), (128,128,128)
]

def get_color(idx):
    return color_map[idx % len(color_map)] if idx != -1 else (0,0,0)

def load_images(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(SUPPORTED_EXTS)]

embeddings = []
face_infos = []
img_paths = load_images(IMAGE_FOLDER)
print(f"Loaded {len(img_paths)} images.")

# --------- Embedding extraction ----------
for img_path in img_paths:
    try:
        img_pil = Image.open(img_path)
        img = np.array(img_pil)

        # Handle grayscale / RGBA
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Convert RGB (PIL) -> BGR (OpenCV/InsightFace)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    except Exception as e:
        print(f"Could not load {img_path}: {e}")
        continue

    faces = app.get(img)
    for face in faces:
        embeddings.append(face.embedding)
        bbox = face.bbox.astype(int)
        face_infos.append({
            'img_path': img_path,
            'bbox': bbox,
            'embedding_idx': len(embeddings) - 1,
            'embedding': face.embedding
        })

embeddings = np.array(embeddings)
print(f"Extracted {len(embeddings)} face embeddings.")

# --------- Clustering ----------
eps = 0.35
db = DBSCAN(eps=eps, min_samples=2, metric='cosine')
labels = db.fit_predict(embeddings)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f'Clusters found: {n_clusters}, Noise faces: {(labels == -1).sum()}')

# --------- Cluster-wise export ----------
cluster_map = {}
for info, label in zip(face_infos, labels):
    cluster_map.setdefault(label, []).append(info)

for clus_id, infos in cluster_map.items():
    clus_name = f"cluster_{clus_id}" if clus_id != -1 else "noise"
    clus_folder = os.path.join(EXPORT_DIR, clus_name)
    os.makedirs(clus_folder, exist_ok=True)

    img_face_map = {}
    for info in infos:
        img_face_map.setdefault(info['img_path'], []).append(info['bbox'])

    for img_path, bboxes in img_face_map.items():
        try:
            img_pil = Image.open(img_path)
            img = np.array(img_pil)

            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            # RGB -> BGR before drawing and saving
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        except Exception as e:
            print(f"Could not load {img_path} for marking: {e}")
            continue

        color = get_color(clus_id)
        for box in bboxes:
            x1, y1, x2, y2 = box
            cluster_label = f'Cluster {clus_id}' if clus_id != -1 else "Noise"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                img, cluster_label, (x1, y1-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA
            )

        out_base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(clus_folder, out_base + ".png")
        cv2.imwrite(out_path, img)

print(f"Cluster-wise folders with marked images saved to {EXPORT_DIR}!")

# --------- Save face info for later search ----------
faceinfos_to_save = []
for info, label in zip(face_infos, labels):
    faceinfos_to_save.append({
        'img_path': info['img_path'],
        'embedding': info['embedding'],
        'cluster_label': label,
        'bbox': info['bbox'].tolist()
    })

with open("faceinfos.pkl", "wb") as f:
    pickle.dump(faceinfos_to_save, f)

print("Saved faceinfos.pkl for search!")
