import argparse, csv, base64, io
import os
from pathlib import Path

import openai
from PIL import Image

# Load API key from environment variable for security
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set via environment variable

ATTRS = [
    "muscular_male",
    "sexualized_female",
    "bright_vibrant_colors",
    "scary_villain",
]

SYSTEM_PROMPT = (
    "You are an art critic scoring comic-book covers. "
    "For each image assign an integer score from 0 (no evidence) to 5 (strong evidence) "
    f"for the following attributes: {', '.join(ATTRS)}. "
    'Return ONLY a JSON object exactly like: '
    '{"muscular_male":<int>, "sexualized_female":<int>, '
    '"bright_vibrant_colors":<int>, "scary_villain":<int>}'
)

def img_to_data_uri(path: Path) -> str:
    """Encode image as base64 data URI (avoids hosting)."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"

def gpt_score(image_path: Path, model="gpt-4o-mini"):
    """Send one image → JSON scores."""
    uri = img_to_data_uri(image_path)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": "Here is the cover to score:"},
            {"type": "image_url", "image_url": {"url": uri}}
        ]}
    ]
    resp = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", default="comic_images")
    ap.add_argument("--out_csv", default="gpt_stimuli_scores.csv")
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    images = sorted([p for p in img_dir.iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
    

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file"] + ATTRS)

        for img in images:
            scores = gpt_score(img, model=args.model)
            row = [img.name] + [eval(scores)[k] for k in ATTRS]
            writer.writerow(row)
            print(f"{img.name:<35}  {scores}")

if __name__ == "__main__":
    main()
