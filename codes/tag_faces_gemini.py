import os
import json
import pickle
import time
import re
import pillow_heif
pillow_heif.register_heif_opener()

from PIL import Image
from tqdm import tqdm
from typing import List

from google import genai
from google.genai import types

# =========================
# CONFIG
# =========================
BASE_DIR = "/Users/pranjalgarg/Desktop/cv_project"
DATA_DIR = os.path.join(BASE_DIR, "data")
FACEINFOS_PATH = os.path.join(DATA_DIR, "faceinfos.pkl")
OUT_JSONL = os.path.join(DATA_DIR, "records.jsonl")

MODEL_ID = "gemini-2.5-flash"
SLEEP_BETWEEN_CALLS = 0.25

# Expand bbox to include upper body for clothing tags
X_EXPAND = 0.35
Y_UP_EXPAND = 0.10
Y_DOWN_EXPAND = 1.60

MAX_TAGS = 10


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

def get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Run: export GEMINI_API_KEY='YOUR_KEY'")
    return genai.Client(api_key=api_key)

def response_to_text(resp) -> str:
    # 1) resp.text
    txt = getattr(resp, "text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    # 2) candidates[0].content.parts[].text
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

    # Strip code fences if Gemini uses them
    if "```" in text:
        lines = [l.strip() for l in text.splitlines()
                 if l.strip() and not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Split into lines; also handle comma-separated
    lines = [l.strip(" -•\t").strip() for l in text.splitlines() if l.strip()]
    if len(lines) == 1 and "," in lines[0]:
        lines = [x.strip() for x in lines[0].split(",")]

    # Filter junk
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

    # De-dup preserve order
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            uniq.append(t)
            seen.add(t)

    return uniq[:MAX_TAGS]


def main():
    if not os.path.exists(FACEINFOS_PATH):
        raise FileNotFoundError(f"Missing {FACEINFOS_PATH}. Run cluster_faces.py first.")

    os.makedirs(DATA_DIR, exist_ok=True)

    client = get_client()

    with open(FACEINFOS_PATH, "rb") as f:
        face_records = pickle.load(f)

    print(f"Loaded {len(face_records)} face records.")

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

    written = 0
    empty = 0
    fail_prints_left = 3

    with open(OUT_JSONL, "a", encoding="utf-8") as out_f:
        for rec in tqdm(face_records, desc="Tagging with Gemini"):
            rid = rec.get("id")
            img_path = rec.get("img_path")
            bbox = rec.get("bbox")
            person_id = int(rec.get("cluster_label", -1))

            if not rid or not img_path or bbox is None or not os.path.exists(img_path):
                continue

            try:
                w, h = Image.open(img_path).size
                crop_box = expand_bbox(bbox, w, h)
                jpeg_bytes = crop_to_jpeg_bytes(img_path, crop_box)
            except:
                continue

            # ✅ THIS is the key fix: send image as SDK Part (not a dict)
            img_part = types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")

            try:
                resp = client.models.generate_content(
                    model=MODEL_ID,
                    contents=[prompt, img_part],
                )
                text = response_to_text(resp)
                tags = parse_tags_plain(text)
            except Exception as e:
                if fail_prints_left > 0:
                    print("Gemini error:", repr(e))
                    fail_prints_left -= 1
                tags = []

            if not tags:
                empty += 1
                continue

            out_obj = {
                "id": rid,
                "person_id": person_id,
                "img_path": img_path,
                "face_bbox": bbox,
                "crop_bbox": list(crop_box),
                "tags": tags,
            }
            out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            out_f.flush()

            written += 1
            time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"Done. Written={written}, empty={empty}")
    print(f"Saved tags to: {OUT_JSONL}")


if __name__ == "__main__":
    main()
