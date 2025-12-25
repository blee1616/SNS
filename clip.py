import os, csv, argparse, torch
import open_clip
from PIL import Image
from pathlib import Path

PROMPTS = [
    "a muscular superhero cartoon in an intense action pose",
    "a highly sexualized female cartoon front and center",
    "an explosive action scene with bright, vibrant colors",
    "a terrifying monstrous villain attacking violently",
    
]

NEG_PROMPT = "a plain, normal comic cover with no exaggerated features"

def load_model(device="cpu"):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model = model.to(device)
    model.eval()
    return model, preprocess

def score_image(model, preprocess, image_path, device="cpu"):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    text_tokens = open_clip.tokenize(PROMPTS + [NEG_PROMPT]).to(device)

    with torch.no_grad():
        img_emb = model.encode_image(image)
        txt_emb = model.encode_text(text_tokens)

    img_emb /= img_emb.norm(dim=-1, keepdim=True)
    txt_emb /= txt_emb.norm(dim=-1, keepdim=True)

    sims = (img_emb @ txt_emb.T).squeeze(0).cpu().numpy()

    pos_scores = sims[:-1]
    neg_score = sims[-1]
    stimuli_index = sum(pos_scores) - neg_score

    return pos_scores, neg_score, stimuli_index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default="covers", help="Folder of cover images")
    parser.add_argument("--out_csv", default="clip_stimuli_scores.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, preprocess = load_model(args.device)

    img_dir = Path(args.img_dir)
    image_files = [p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["file"] + [f"prompt_{i+1}" for i in range(len(PROMPTS))] + ["plain_score", "stimuli_index"]
        writer.writerow(header)

        for img_path in image_files:
            pos_scores, neg_score, stim_idx = score_image(model, preprocess, img_path, args.device)
            writer.writerow([img_path.name, *pos_scores, neg_score, stim_idx])
            print(f"{img_path.name:40s}  StimuliIndex={stim_idx:6.3f}")

if __name__ == "__main__":
    main()
