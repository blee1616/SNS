import pandas as pd
import numpy as np
import re
import csv
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import open_clip
from PIL import Image
import json
import openai
import base64
import io
import argparse

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_icv2_data(csv_file):
    """Load ICV2 comics data and extract revenue from units column"""
    comics_data = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rank = 1
        for row in reader:
            if len(row) >= 5:
                # Parse the data: Revenue, Title, Price, Publisher, Units (with revenue in quotes)
                revenue_index = row[0]
                title = row[1] 
                price = row[2]
                publisher = row[3]
                units_with_revenue = row[4].replace('"', '').replace(',', '')
                
                try:
                    revenue = float(units_with_revenue)
                except ValueError:
                    revenue = 0.0
                
                comics_data.append({
                    'rank': rank,
                    'title': title,
                    'price': price,
                    'publisher': publisher,
                    'revenue': revenue,
                    'revenue_index': revenue_index
                })
                rank += 1
    
    return pd.DataFrame(comics_data)

def load_clip_model(device="cpu"):
    """Load OpenCLIP model"""
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model, preprocess, tokenizer

def analyze_image_clip(model, preprocess, tokenizer, image_path, device="cpu"):
    """Analyze single image with CLIP model"""
    # Define prompts for supernormal stimuli
    prompts = [
        "a muscular superhero cartoon in an intense action pose",
        "a highly sexualized female cartoon front and center", 
        "an explosive action scene with bright, vibrant colors",
        "a terrifying monstrous villain attacking violently"
    ]
    negative_prompt = "a plain, normal comic cover with no exaggerated features"
    
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    
    # Tokenize all prompts
    all_prompts = prompts + [negative_prompt]
    text_inputs = tokenizer(all_prompts).to(device)
    
    # Get embeddings
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
        
        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarities
        similarities = (image_features @ text_features.T).cpu().numpy()[0]
    
    # Calculate stimuli index as SUM of positives minus negative
    positive_scores = similarities[:-1]  # First 4 scores
    negative_score = similarities[-1]    # Last score
    stimuli_index = float(np.sum(positive_scores) - negative_score)
    
    return {
        'prompt_scores': positive_scores.tolist(),
        'negative_score': float(negative_score),
        'stimuli_index': float(stimuli_index),
        'individual_scores': {
            'muscular_male': float(positive_scores[0]),
            'sexualized_female': float(positive_scores[1]),
            'bright_colors': float(positive_scores[2]),
            'scary_villain': float(positive_scores[3])
        }
    }

def img_to_data_uri(image_path):
    """Convert image to base64 data URI for GPT-4"""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"

def analyze_image_gpt4(image_path, api_key):
    """Analyze single image with GPT-4 Vision"""
    openai.api_key = api_key
    
    system_prompt = (
        "You are an art critic scoring comic-book covers. "
        "For each image assign an integer score from 0 (no evidence) to 5 (strong evidence) "
        "for the following attributes: muscular_male, sexualized_female, bright_vibrant_colors, scary_villain. "
        'Return ONLY a JSON object exactly like: '
        '{"muscular_male":<int>, "sexualized_female":<int>, '
        '"bright_vibrant_colors":<int>, "scary_villain":<int>}'
    )
    
    try:
        data_uri = img_to_data_uri(image_path)
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "Please analyze this comic cover:"},
                        {"type": "image_url", "image_url": {"url": data_uri}}
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=150
        )
        
        # Parse JSON response
        result = json.loads(response.choices[0].message.content)
        
        # Calculate stimuli index using MAX instead of SUM
        stimuli_index = max(result.values())  # CHANGED: max instead of sum
        
        return {
            'scores': result,
            'stimuli_index': stimuli_index,
            'muscular_male': result.get('muscular_male', 0),
            'sexualized_female': result.get('sexualized_female', 0),
            'bright_vibrant_colors': result.get('bright_vibrant_colors', 0),
            'scary_villain': result.get('scary_villain', 0)
        }
        
    except Exception as e:
        print(f"Error analyzing {image_path} with GPT-4: {e}")
        return None

def process_image_folder(folder_path, model, preprocess, tokenizer, device, gpt_api_key=None, min_rank: int = 1, max_rank: int | None = None):
    """Process all images in a folder with both CLIP and GPT-4. Optionally filter by rank range."""
    folder = Path(folder_path)
    results = []
    
    # Get all image files and sort them numerically (case-insensitive)
    image_files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}
    ]
    
    # Sort by numeric filename
    def get_rank_from_filename(filepath):
        try:
            return int(filepath.stem)
        except ValueError:
            return float('inf')  # Put non-numeric files at end
    
    image_files.sort(key=get_rank_from_filename)
    
    print(f"\nProcessing {len(image_files)} images from {folder_path}...")
    
    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing {img_path.name}...")
        
        # Get rank from filename
        try:
            rank = int(img_path.stem)
        except ValueError:
            print(f"Warning: Could not parse rank from {img_path.name}")
            continue
        
        # Apply rank filter
        if rank < min_rank or (max_rank is not None and rank > max_rank):
            continue

        # CLIP analysis
        clip_result = analyze_image_clip(model, preprocess, tokenizer, img_path, device)
        
        # GPT-4 analysis (if API key provided)
        gpt_result = None
        if gpt_api_key:
            gpt_result = analyze_image_gpt4(img_path, gpt_api_key)
        
        if clip_result:
            result = {
                'rank': rank,
                'filename': img_path.name,
                'clip_stimuli_index': clip_result['stimuli_index'],
                'clip_muscular_male': clip_result['individual_scores']['muscular_male'],
                'clip_sexualized_female': clip_result['individual_scores']['sexualized_female'],
                'clip_bright_colors': clip_result['individual_scores']['bright_colors'],
                'clip_scary_villain': clip_result['individual_scores']['scary_villain'],
                'clip_negative_score': clip_result['negative_score']
            }
            
            if gpt_result:
                result.update({
                    'gpt_stimuli_index': gpt_result['stimuli_index'],
                    'gpt_muscular_male': gpt_result['muscular_male'],
                    'gpt_sexualized_female': gpt_result['sexualized_female'],
                    'gpt_bright_colors': gpt_result['bright_vibrant_colors'],
                    'gpt_scary_villain': gpt_result['scary_villain']
                })
            
            results.append(result)
    
    return pd.DataFrame(results)

def merge_with_icv2_data(analysis_df, icv2_df):
    """Merge analysis results with ICV2 revenue data"""
    # Merge on rank
    merged_df = pd.merge(analysis_df, icv2_df, on='rank', how='inner')
    return merged_df

def run_regression_analysis(df, dataset_name):
    """Run comprehensive regression analysis"""
    print(f"\n{'='*60}")
    print(f"REGRESSION ANALYSIS - {dataset_name}")
    print(f"{'='*60}")
    print(f"Sample size: {len(df)} comics")
    print(f"Revenue range: ${df['revenue'].min():,.0f} - ${df['revenue'].max():,.0f}")
    print()
    
    results = {}
    
    # Define variables to test
    variables_to_test = [
        ('clip_stimuli_index', 'CLIP Stimuli Index'),
        ('clip_muscular_male', 'CLIP Muscular Male'),
        ('clip_sexualized_female', 'CLIP Sexualized Female'),
        ('clip_bright_colors', 'CLIP Bright Colors'),
        ('clip_scary_villain', 'CLIP Scary Villain')
    ]
    
    # Add GPT variables if they exist
    if 'gpt_stimuli_index' in df.columns:
        variables_to_test.extend([
            ('gpt_stimuli_index', 'GPT-4 Stimuli Index'),
            ('gpt_muscular_male', 'GPT-4 Muscular Male'),
            ('gpt_sexualized_female', 'GPT-4 Sexualized Female'),
            ('gpt_bright_colors', 'GPT-4 Bright Colors'),
            ('gpt_scary_villain', 'GPT-4 Scary Villain')
        ])
    
    for var_name, display_name in variables_to_test:
        if var_name not in df.columns:
            continue
            
        print(f"{display_name} vs Revenue:")
        print("-" * 50)
        
        # Prepare data
        X = df[[var_name]].values
        y = df['revenue'].values
        
        # Run regression
        reg = LinearRegression()
        reg.fit(X, y)
        y_pred = reg.predict(X)
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        correlation = np.corrcoef(df[var_name], df['revenue'])[0, 1]
        
        print(f"R² Score: {r2:.4f}")
        print(f"Correlation: {correlation:.4f}")
        print(f"MSE: {mse:.2f}")
        print(f"Coefficient: {reg.coef_[0]:.2f}")
        print(f"Intercept: {reg.intercept_:.2f}")
        print()
        
        results[var_name] = {
            'r2': r2,
            'correlation': correlation,
            'mse': mse,
            'coefficient': reg.coef_[0],
            'intercept': reg.intercept_,
            'display_name': display_name
        }
    
    return results

def create_visualizations(df, results, dataset_name):
    """Create and return two separate figures: regression and correlations (no GPT, no revenue distribution)."""
    figs = {}

    # Regression: CLIP Stimuli Index vs Revenue
    if 'clip_stimuli_index' in df.columns:
        fig_reg, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df['clip_stimuli_index'], df['revenue'], alpha=0.7, s=50)

        # Regression line
        X = df[['clip_stimuli_index']].values
        y = df['revenue'].values
        reg = LinearRegression().fit(X, y)
        x_line = np.linspace(df['clip_stimuli_index'].min(), df['clip_stimuli_index'].max(), 100)
        y_line = reg.predict(x_line.reshape(-1, 1))
        ax.plot(x_line, y_line, 'r--', alpha=0.8, linewidth=2)

        ax.set_xlabel('CLIP Stimuli Index')
        ax.set_ylabel('Revenue')
        ax.set_title(f'{dataset_name} - CLIP Stimuli Index vs Revenue\nR² = {results.get("clip_stimuli_index", {}).get("r2", 0):.3f}')
        ax.grid(True, alpha=0.3)
        fig_reg.tight_layout()
        figs['regression'] = fig_reg

    # Correlations: Individual CLIP features
    features = ['clip_muscular_male', 'clip_sexualized_female', 'clip_bright_colors', 'clip_scary_villain']
    feature_names = ['Muscular\nMale', 'Sexualized\nFemale', 'Bright\nColors', 'Scary\nVillain']
    correlations = [results.get(feat, {}).get('correlation', 0) for feat in features]

    # Make Oct 2015 chart larger to avoid label overlap
    corr_figsize = (12, 8) if ('oct' in dataset_name.lower()) else (8, 6)
    fig_corr, ax = plt.subplots(figsize=corr_figsize)
    bars = ax.bar(feature_names, correlations, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylabel('Correlation with Revenue')
    ax.set_title(f'{dataset_name} - Individual CLIP Feature Correlations')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + (0.02 if height >= 0 else -0.05),
            f'{corr:.3f}',
            ha='center',
            va='bottom' if height >= 0 else 'top',
            fontsize=12 if ('oct' in dataset_name.lower()) else 10
        )
    fig_corr.tight_layout()
    figs['correlations'] = fig_corr

    return figs

def main():
    parser = argparse.ArgumentParser(description='Analyze ICV2 comic data with CLIP and GPT-4')
    parser.add_argument('--jan2020_folder', default='Jan 2020', help='Folder containing Jan 2020 comic images')
    parser.add_argument('--mar2020_folder', default='Mar 2020', help='Folder containing Mar 2020 comic images')
    parser.add_argument('--oct2015_folder', default='OCT 2015', help='Folder containing Oct 2015 comic images')
    parser.add_argument('--jan2020_csv', default='ICV2 comics - Jan 2020.csv', help='Jan 2020 ICV2 data CSV')
    parser.add_argument('--mar2020_csv', default='ICV2 comics - Mar 2020.csv', help='Mar 2020 ICV2 data CSV')
    parser.add_argument('--oct2015_csv', default='ICV2 comics - Oct 2015.csv', help='Oct 2015 ICV2 data CSV')
    parser.add_argument('--gpt_api_key', default=None, help='OpenAI API key for GPT-4 analysis')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--skip_gpt', action='store_true', help='Skip GPT-4 analysis to save costs')
    parser.add_argument('--min_rank', type=int, default=1, help='Minimum rank to include (inclusive)')
    parser.add_argument('--max_rank', type=int, default=None, help='Maximum rank to include (inclusive)')
    args = parser.parse_args()
    
    print("🎨 ICV2 Comics Supernormal Stimuli Analysis")
    print("=" * 50)
    
    # Load CLIP model
    print("Loading CLIP model...")
    model, preprocess, tokenizer = load_clip_model(args.device)
    print(f"Using device: {args.device}")
    
    # Process Jan 2020 data
    print("\n📊 PROCESSING JAN 2020 DATA")
    print("-" * 40)
    
    if Path(args.jan2020_folder).exists() and Path(args.jan2020_csv).exists():
        # Load ICV2 data
        jan2020_icv2 = load_icv2_data(args.jan2020_csv)
        print(f"Loaded {len(jan2020_icv2)} comics from Jan 2020 ICV2 data")
        
        # Process images
        gpt_key = None if args.skip_gpt else args.gpt_api_key
        jan2020_analysis = process_image_folder(
            args.jan2020_folder, model, preprocess, tokenizer, args.device, gpt_key,
            min_rank=args.min_rank, max_rank=args.max_rank
        )
        
        # Merge with ICV2 data
        jan2020_merged = merge_with_icv2_data(jan2020_analysis, jan2020_icv2)
        print(f"Successfully matched {len(jan2020_merged)} comics with revenue data")
        
        # Run regression analysis
        jan2020_results = run_regression_analysis(jan2020_merged, "Jan 2020")
        
        # Create visualizations (separate figures)
        figs1 = create_visualizations(jan2020_merged, jan2020_results, "Jan 2020")
        figs1['regression'].savefig('jan2020_regression.png', dpi=300, bbox_inches='tight')
        figs1['correlations'].savefig('jan2020_correlations.png', dpi=300, bbox_inches='tight')
        
        # Save detailed results
        jan2020_merged.to_csv('jan2020_detailed_results.csv', index=False)
        
    else:
        print("❌ Jan 2020 folder or CSV not found")
    
    # Process Mar 2020 data
    print("\n📊 PROCESSING MAR 2020 DATA")
    print("-" * 40)
    
    if Path(args.mar2020_folder).exists() and Path(args.mar2020_csv).exists():
        # Load ICV2 data
        mar2020_icv2 = load_icv2_data(args.mar2020_csv)
        print(f"Loaded {len(mar2020_icv2)} comics from Mar 2020 ICV2 data")
        
        # Process images
        gpt_key = None if args.skip_gpt else args.gpt_api_key
        mar2020_analysis = process_image_folder(
            args.mar2020_folder, model, preprocess, tokenizer, args.device, gpt_key,
            min_rank=args.min_rank, max_rank=args.max_rank
        )
        
        # Merge with ICV2 data
        mar2020_merged = merge_with_icv2_data(mar2020_analysis, mar2020_icv2)
        print(f"Successfully matched {len(mar2020_merged)} comics with revenue data")
        
        # Run regression analysis
        mar2020_results = run_regression_analysis(mar2020_merged, "Mar 2020")
        
        # Create visualizations (separate figures)
        figs2 = create_visualizations(mar2020_merged, mar2020_results, "Mar 2020")
        figs2['regression'].savefig('mar2020_regression.png', dpi=300, bbox_inches='tight')
        figs2['correlations'].savefig('mar2020_correlations.png', dpi=300, bbox_inches='tight')
        
        # Save detailed results
        mar2020_merged.to_csv('mar2020_detailed_results.csv', index=False)
        
    else:
        print("❌ Mar 2020 folder or CSV not found")

    # Process Oct 2015 data
    print("\n📊 PROCESSING OCT 2015 DATA")
    print("-" * 40)
    if Path(args.oct2015_folder).exists() and Path(args.oct2015_csv).exists():
        # Load ICV2 data
        oct2015_icv2 = load_icv2_data(args.oct2015_csv)
        print(f"Loaded {len(oct2015_icv2)} comics from Oct 2015 ICV2 data")

        # Process images
        gpt_key = None if args.skip_gpt else args.gpt_api_key
        oct2015_analysis = process_image_folder(
            args.oct2015_folder, model, preprocess, tokenizer, args.device, gpt_key,
            min_rank=args.min_rank, max_rank=args.max_rank
        )

        # Merge with ICV2 data
        oct2015_merged = merge_with_icv2_data(oct2015_analysis, oct2015_icv2)
        print(f"Successfully matched {len(oct2015_merged)} comics with revenue data")

        # Run regression analysis
        oct2015_results = run_regression_analysis(oct2015_merged, "Oct 2015")

        # Create visualizations (separate figures)
        figs_oct = create_visualizations(oct2015_merged, oct2015_results, "Oct 2015")
        figs_oct['regression'].savefig('oct2015_regression.png', dpi=300, bbox_inches='tight')
        figs_oct['correlations'].savefig('oct2015_correlations.png', dpi=300, bbox_inches='tight')

        # Save detailed results
        oct2015_merged.to_csv('oct2015_detailed_results.csv', index=False)
    else:
        print("❌ Oct 2015 folder or CSV not found")
    
    # Combined analysis (combine any available datasets)
    datasets = []
    if 'jan2020_merged' in locals():
        jan2020_merged['dataset'] = 'Jan 2020'
        datasets.append(jan2020_merged)
    if 'mar2020_merged' in locals():
        mar2020_merged['dataset'] = 'Mar 2020'
        datasets.append(mar2020_merged)
    if 'oct2015_merged' in locals():
        oct2015_merged['dataset'] = 'Oct 2015'
        datasets.append(oct2015_merged)

    if len(datasets) >= 2:
        print("\n📊 COMBINED ANALYSIS")
        print("-" * 40)
        
        # Combine datasets
        combined_df = pd.concat(datasets, ignore_index=True)
        
        print(f"Combined dataset: {len(combined_df)} comics")
        
        # Run combined analysis
        combined_results = run_regression_analysis(combined_df, "Combined")
        
        # Create combined visualizations (separate figures)
        figs3 = create_visualizations(combined_df, combined_results, "Combined Analysis")
        figs3['regression'].savefig('combined_regression.png', dpi=300, bbox_inches='tight')
        figs3['correlations'].savefig('combined_correlations.png', dpi=300, bbox_inches='tight')
        
        # Save combined results
        combined_df.to_csv('combined_detailed_results.csv', index=False)
        
        # Create summary report
        print("\n📋 SUMMARY REPORT")
        print("=" * 50)
        print(f"Total comics analyzed: {len(combined_df)}")
        print(f"CLIP Stimuli Index - Revenue correlation: {combined_results.get('clip_stimuli_index', {}).get('correlation', 'N/A'):.4f}")
        if 'gpt_stimuli_index' in combined_results:
            print(f"GPT-4 Stimuli Index - Revenue correlation: {combined_results.get('gpt_stimuli_index', {}).get('correlation', 'N/A'):.4f}")
        
        print(f"\nStrongest individual correlations:")
        for var, data in combined_results.items():
            if 'correlation' in data:
                print(f"  {data['display_name']}: {data['correlation']:.4f}")
    
    plt.show()
    print("\n✅ Analysis complete! Check the generated CSV files and PNG visualizations.")

if __name__ == "__main__":
    main()