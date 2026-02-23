import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import insightface
from insightface.app import FaceAnalysis

# Initialize ArcFace
app = FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

IMAGE_FOLDER = '/Users/pranjalgarg/Desktop/test_image_recogo/small_data'  # your dataset path
EXPORT_DIR = '/Users/pranjalgarg/Desktop/test_image_recogo/arcface_clusters_export'
os.makedirs(EXPORT_DIR, exist_ok=True)

# Color map for clusters (BGR)
color_map = [
    (0,0,255),(0,255,0),(255,0,0),(0,165,255),(128,0,128),(255,255,0),
    (255,0,255),(203,192,255),(0,255,255),(50,205,50),(255,255,240),(128,128,128)
]
def get_color(idx):
    return color_map[idx % len(color_map)] if idx != -1 else (0,0,̉0)

# Embedding extraction
embeddings, face_infos = [], []
def load_images(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]
img_paths = load_images(IMAGE_FOLDER)
print(f"Loaded {len(img_paths)} images.")

for img_path in img_paths:
    img = cv2.imread(img_path)
    faces = app.get(img)
    for face in faces:
        embeddings.append(face.embedding)
        bbox = face.bbox.astype(int)
        face_infos.append({'img_path': img_path, 'bbox': bbox, 'embedding_idx': len(embeddings)-1})

embeddings = np.array(embeddings)
print(f"Extracted {len(embeddings)} face embeddings.")

# Clustering
eps = 0.35
db = DBSCAN(eps=eps, min_samples=2, metric='cosine')
labels = db.fit_predict(embeddings)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f'Clusters found: {n_clusters}, Noise faces: {(labels == -1).sum()}')

# Map clusters to face_infos
cluster_map = {}
for info, label in zip(face_infos, labels):
    cluster_map.setdefault(label, []).append(info)

# For each cluster, create folder and save annotated images
for clus_id, infos in cluster_map.items():
    # Cluster folder
    clus_name = f"cluster_{clus_id}" if clus_id != -1 else "noise"
    clus_folder = os.path.join(EXPORT_DIR, clus_name)
    os.makedirs(clus_folder, exist_ok=True)

    # Find images with faces from this cluster
    img_face_map = {}
    for info in infos:
        img_face_map.setdefault(info['img_path'], []).append(info['bbox'])

    # Annotate and save for cluster
    for img_path, bboxes in img_face_map.items():
        img = cv2.imread(img_path)
        color = get_color(clus_id)
        for box in bboxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            label_text = f'Cluster {clus_id}' if clus_id != -1 else 'Noise'
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, label_text, (x1, y1-12), font, 0.9, color, 2, cv2.LINE_AA)
        out_path = os.path.join(clus_folder, os.path.basename(img_path))
        cv2.imwrite(out_path, img)

print(f"Cluster-wise folders with marked images saved to {EXPORT_DIR}!")