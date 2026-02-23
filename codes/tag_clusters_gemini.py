import os
import json
import pickle
import time
import re
import pillow_heif
pillow_heif.register_heif_opener()

from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple

from google import genai
from google.genai import types

# =========================
# CONFIG
# =========================
BASE_DIR = "/Users/pranjalgarg/Desktop/cv_project"
DATA_DIR = os.path.join(BASE_DIR, "data")

FACEINFOS_PATH = os.path.join(DATA_DIR, "faceinfos.pkl")
CLUSTER_TAGS_PATH = os.path.join(DATA_DIR, "cluster_tags.json")
OUT_JSONL = os.path.join(DATA_DIR, "records.jsonl")

MODEL_ID = "gemini-2.5-flash"

# Free tier is limited (often 20/day/model). Keep safe.
MAX_CALLS_PER_RUN = 20

# Expand bbox to include upper body/clothing
X_EXPAND = 0.35
Y_UP_EXPAND = 0.10
Y_DOWN_EXPAND = 1.60

MAX_TAGS = 10
SLEEP_BETWEEN_CALLS = 0.35


# =========================
# Helpers
# =========================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def expand_bbox(bbox, w, h):
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1

    ex1 = int(x1 - X_EXPAND * bw)
    ex2 = int(x2 + X_EXPAND * bw)
    ey1 = int(y1 - Y_UP_EXPAND * bh)
    ey2 = int(y2 + Y_DOWN_EXPAND * bh)

    ex1 = clamp(ex1, 0, w - 1)
    ex2 = clamp(ex2, 1, w)
    ey1 = clamp(ey1, 0, h - 1)
    ey2 = clamp(ey2, 1, h)

    if ex2 <= ex1 + 2 or ey2 <= ey1 + 2:
        return (x1, y1, x2, y2)

    return (ex1, ey1, ex2, ey2)

def crop_to_jpeg_bytes(img_path: str, crop_box) -> bytes:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    x1, y1, x2, y2 = crop_box
    x1 = clamp(x1, 0, w - 1)
    x2 = clamp(x2, 1, w)
    y1 = clamp(y1, 0, h - 1)
    y2 = clamp(y2, 1, h)

    crop = img.crop((x1, y1, x2, y2))
    crop.thumbnail((512, 512))

    import io
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

def normalize_tag(t: str) -> str:
    t = (t or "").strip().lower()
    t = " ".join(t.split())
    return t

def response_to_text(resp) -> str:
    txt = getattr(resp, "text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    cands = getattr(resp, "candidates", None)
    if cands:
        content = getattr(cands[0], "content", None)
        parts = getattr(content, "parts", None) if content else None
        if parts:
            out = []
            for p in parts:
                t = getattr(p, "text", None)
                if isinstance(t, str) and t.strip():
                    out.append(t.strip())
            if out:
                return "\n".join(out).strip()

    return ""

def parse_tags_plain(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    # Remove code fences
    if "```" in text:
        lines = [l.strip() for l in text.splitlines()
                 if l.strip() and not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Split lines, allow comma-separated single line
    lines = [l.strip(" -•\t").strip() for l in text.splitlines() if l.strip()]
    if len(lines) == 1 and "," in lines[0]:
        lines = [x.strip() for x in lines[0].split(",")]

    bad_exact = {"man", "woman", "male", "female", "person", "human", "face"}
    out = []
    for l in lines:
        l = normalize_tag(l)
        if not l:
            continue
        if l in bad_exact:
            continue
        if "{" in l or "}" in l:
            continue
        if re.fullmatch(r"\d+[\).\s]*", l):
            continue
        out.append(l)

    # Dedup preserve order
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            uniq.append(t)
            seen.add(t)

    return uniq[:MAX_TAGS]

def get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Run: export GEMINI_API_KEY='YOUR_KEY'")
    return genai.Client(api_key=api_key)

def load_cluster_tags() -> Dict[str, List[str]]:
    if not os.path.exists(CLUSTER_TAGS_PATH):
        return {}
    with open(CLUSTER_TAGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_cluster_tags(tags_map: Dict[str, List[str]]):
    with open(CLUSTER_TAGS_PATH, "w", encoding="utf-8") as f:
        json.dump(tags_map, f, ensure_ascii=False, indent=2)

def bbox_area(bbox) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)

def pick_representative(face_recs: List[dict]) -> dict:
    return max(face_recs, key=lambda r: bbox_area(r["bbox"]))

def build_records_jsonl(face_records: List[dict], cluster_tags: Dict[str, List[str]]):
    with open(OUT_JSONL, "w", encoding="utf-8") as out_f:
        for r in face_records:
            pid = str(int(r["cluster_label"]))
            tags = cluster_tags.get(pid, [])
            obj = {
                "id": r["id"],
                "person_id": int(r["cluster_label"]),
                "img_path": r["img_path"],
                "face_bbox": r["bbox"],
                "tags": tags,
            }
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(FACEINFOS_PATH):
        raise FileNotFoundError(f"Missing {FACEINFOS_PATH}. Run cluster_faces.py first.")

    client = get_client()

    with open(FACEINFOS_PATH, "rb") as f:
        face_records = pickle.load(f)

    # Group by cluster/person_id
    clusters: Dict[int, List[dict]] = {}
    for r in face_records:
        pid = int(r["cluster_label"])
        clusters.setdefault(pid, []).append(r)

    # Load existing cluster tags
    cluster_tags = load_cluster_tags()

    # ✅ Only skip clusters that have NON-EMPTY tags
    already = set(k for k, v in cluster_tags.items() if isinstance(v, list) and len(v) > 0)

    # Optional: skip noise cluster -1
    # (remove this if you want to tag -1 too)
    if "-1" in clusters:
        pass  # keep in dataset, but we won't call Gemini for -1

    cluster_items: List[Tuple[int, List[dict]]] = sorted(
        clusters.items(), key=lambda kv: len(kv[1]), reverse=True
    )

    total_clusters = len(cluster_items)
    done_clusters = len(already)
    pending_clusters = len([pid for pid, _ in cluster_items if str(pid) not in already])

    print(f"Total faces: {len(face_records)}")
    print(f"Total clusters: {total_clusters}")
    print(f"Already tagged (non-empty): {done_clusters}")
    print(f"Pending clusters: {pending_clusters}")
    print(f"Will tag up to: {MAX_CALLS_PER_RUN} clusters this run")

    prompt = f"""
You will be given a cropped image of a person (upper body). Return 3 to {MAX_TAGS} tags.

IMPORTANT:
- Return ONLY the tags.
- ONE tag per line.
- No numbering, no bullets, no markdown, no JSON.

Tag rules:
- clothing top color/type if visible (e.g., "black t-shirt", "white shirt", "red hoodie", "blue jacket")
- accessories if visible ("glasses", "cap", "mask")
- scene/background if obvious ("outdoor", "indoor", "night", "selfie", "group photo")
- avoid identity labels like man/woman/person
"""

    calls = 0
    new_tagged = 0

    for pid, recs in tqdm(cluster_items, desc="Tagging clusters"):
        pid_str = str(pid)

        # Optional skip for noise cluster
        if pid == -1:
            continue

        if pid_str in already:
            continue
        if calls >= MAX_CALLS_PER_RUN:
            break

        rep = pick_representative(recs)
        img_path = rep["img_path"]
        bbox = rep["bbox"]

        if not os.path.exists(img_path):
            # ❌ DO NOT write empty list to cluster_tags (it breaks resume)
            continue

        try:
            w, h = Image.open(img_path).size
            crop_box = expand_bbox(bbox, w, h)
            jpeg_bytes = crop_to_jpeg_bytes(img_path, crop_box)

            img_part = types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")

            resp = client.models.generate_content(
                model=MODEL_ID,
                contents=[prompt, img_part],
            )

            text = response_to_text(resp)
            tags = parse_tags_plain(text)

            # ✅ Save only if we got tags
            if tags:
                cluster_tags[pid_str] = tags
                already.add(pid_str)
                new_tagged += 1
                calls += 1
                save_cluster_tags(cluster_tags)
                time.sleep(SLEEP_BETWEEN_CALLS)
            else:
                # if empty tags, just skip (don’t mark as done)
                continue

        except Exception as e:
            msg = repr(e)

            # Stop immediately on quota
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                print("\nQuota hit (429). Stopping now. Run again tomorrow or enable billing.")
                break

            print(f"\nError tagging cluster {pid}: {msg}")
            # do NOT mark empty as done
            continue

    # Build per-face records.jsonl using cluster tags (some may be empty)
    build_records_jsonl(face_records, cluster_tags)

    print("\n========== SUMMARY ==========")
    print(f"Gemini calls made this run: {calls}")
    print(f"Clusters with non-empty tags this run: {new_tagged}")
    print(f"Saved cluster tags: {CLUSTER_TAGS_PATH}")
    print(f"Rebuilt records file ({len(face_records)} faces): {OUT_JSONL}")
    print("================================\n")


if __name__ == "__main__":
    main()
