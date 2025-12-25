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
from scipy import stats
from scipy.stats import t

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
    # Define prompts for supernormal stimuli - REMOVED sexualization prompt
    prompts = [
        "a muscular superhero cartoon in an intense action pose",
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
    positive_scores = similarities[:-1]  # First 3 scores (removed sexualization)
    negative_score = similarities[-1]    # Last score
    stimuli_index = float(np.sum(positive_scores) - negative_score)
    
    return {
        'prompt_scores': positive_scores.tolist(),
        'negative_score': float(negative_score),
        'stimuli_index': float(stimuli_index),
        'individual_scores': {
            'muscular_male': float(positive_scores[0]),
            'bright_colors': float(positive_scores[1]),
            'scary_villain': float(positive_scores[2])
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
    
    # Updated system prompt without sexualization
    system_prompt = (
        "You are an art critic scoring comic-book covers. "
        "For each image assign an integer score from 0 (no evidence) to 5 (strong evidence) "
        "for the following attributes: muscular_male, bright_vibrant_colors, scary_villain. "
        'Return ONLY a JSON object exactly like: '
        '{"muscular_male":<int>, "bright_vibrant_colors":<int>, "scary_villain":<int>}'
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
        
        # Calculate stimuli index using SUM instead of MAX for consistency with CLIP
        stimuli_index = sum(result.values())
        
        return {
            'scores': result,
            'stimuli_index': stimuli_index,
            'muscular_male': result.get('muscular_male', 0),
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
                'clip_bright_colors': clip_result['individual_scores']['bright_colors'],
                'clip_scary_villain': clip_result['individual_scores']['scary_villain'],
                'clip_negative_score': clip_result['negative_score']
            }
            
            if gpt_result:
                result.update({
                    'gpt_stimuli_index': gpt_result['stimuli_index'],
                    'gpt_muscular_male': gpt_result['muscular_male'],
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

def calculate_comprehensive_statistics(X, y, model_name):
    """Calculate comprehensive regression statistics including β̂, 95% CI, p-value, R², and ΔR²"""
    n = len(X)  # Sample size
    X_with_intercept = np.column_stack([np.ones(n), X])  # Add intercept column
    
    # Fit the model
    reg = LinearRegression()
    reg.fit(X.reshape(-1, 1), y)
    y_pred = reg.predict(X.reshape(-1, 1))
    
    # Basic statistics
    beta_0 = reg.intercept_  # Intercept
    beta_1 = reg.coef_[0]    # Slope coefficient (β̂)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    correlation = np.corrcoef(X, y)[0, 1]
    
    # Calculate standard errors and t-statistics
    residuals = y - y_pred
    residual_sum_squares = np.sum(residuals ** 2)
    degrees_freedom = n - 2  # n - k - 1, where k=1 for simple regression
    
    # Standard error of the regression
    s_yx = np.sqrt(residual_sum_squares / degrees_freedom)
    
    # Standard error of the slope coefficient
    x_mean = np.mean(X)
    sum_x_squared_deviations = np.sum((X - x_mean) ** 2)
    se_beta_1 = s_yx / np.sqrt(sum_x_squared_deviations)
    
    # Standard error of the intercept
    se_beta_0 = s_yx * np.sqrt(1/n + (x_mean**2) / sum_x_squared_deviations)
    
    # t-statistics
    t_stat_beta_1 = beta_1 / se_beta_1
    t_stat_beta_0 = beta_0 / se_beta_0
    
    # p-values (two-tailed test)
    p_value_beta_1 = 2 * (1 - t.cdf(abs(t_stat_beta_1), degrees_freedom))
    p_value_beta_0 = 2 * (1 - t.cdf(abs(t_stat_beta_0), degrees_freedom))
    
    # 95% Confidence intervals
    t_critical = t.ppf(0.975, degrees_freedom)  # 97.5% for two-tailed 95% CI
    
    ci_lower_beta_1 = beta_1 - t_critical * se_beta_1
    ci_upper_beta_1 = beta_1 + t_critical * se_beta_1
    
    ci_lower_beta_0 = beta_0 - t_critical * se_beta_0
    ci_upper_beta_0 = beta_0 + t_critical * se_beta_0
    
    # F-statistic for overall model significance
    mean_y = np.mean(y)
    total_sum_squares = np.sum((y - mean_y) ** 2)
    regression_sum_squares = np.sum((y_pred - mean_y) ** 2)
    f_statistic = (regression_sum_squares / 1) / (residual_sum_squares / degrees_freedom)
    f_p_value = 1 - stats.f.cdf(f_statistic, 1, degrees_freedom)
    
    # ΔR² compared to null model (intercept only)
    null_r2 = 0  # Null model has no predictive power
    delta_r2 = r2 - null_r2
    
    return {
        'model_name': model_name,
        'n': n,
        'beta_0': beta_0,
        'beta_1': beta_1,
        'se_beta_0': se_beta_0,
        'se_beta_1': se_beta_1,
        't_stat_beta_0': t_stat_beta_0,
        't_stat_beta_1': t_stat_beta_1,
        'p_value_beta_0': p_value_beta_0,
        'p_value_beta_1': p_value_beta_1,
        'ci_lower_beta_0': ci_lower_beta_0,
        'ci_upper_beta_0': ci_upper_beta_0,
        'ci_lower_beta_1': ci_lower_beta_1,
        'ci_upper_beta_1': ci_upper_beta_1,
        'r2': r2,
        'delta_r2': delta_r2,
        'correlation': correlation,
        'f_statistic': f_statistic,
        'f_p_value': f_p_value,
        'mse': mse,
        'rmse': np.sqrt(mse),
        'degrees_freedom': degrees_freedom
    }

def run_comprehensive_regression_analysis(df, dataset_name):
    """Run comprehensive regression analysis with advanced statistics"""
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE REGRESSION ANALYSIS - {dataset_name}")
    print(f"{'='*80}")
    print(f"Research Question: Do higher supernormal stimuli scores lead to higher revenue?")
    print(f"Sample size: {len(df)} comics")
    print(f"Revenue range: ${df['revenue'].min():,.0f} - ${df['revenue'].max():,.0f}")
    print()
    
    results = {}
    
    # Focus only on stimuli indices (primary hypothesis)
    variables_to_test = [('clip_stimuli_index', 'CLIP Supernormal Stimuli Index')]
    
    # Add GPT stimuli index if available
    if 'gpt_stimuli_index' in df.columns:
        variables_to_test.append(('gpt_stimuli_index', 'GPT-4 Supernormal Stimuli Index'))
    
    for var_name, display_name in variables_to_test:
        if var_name not in df.columns:
            continue
        
        print(f"\n{display_name} → Revenue Analysis")
        print("=" * 60)
        
        # Get data
        X = df[var_name].values
        y = df['revenue'].values
        
        # Calculate comprehensive statistics
        stats_result = calculate_comprehensive_statistics(X, y, display_name)
        results[var_name] = stats_result
        
        # Print results
        print(f"Model: Revenue = β₀ + β₁ × {display_name} + ε")
        print("-" * 60)
        print(f"Sample Size (n)           : {stats_result['n']}")
        print(f"Degrees of Freedom        : {stats_result['degrees_freedom']}")
        print()
        
        print("COEFFICIENT ESTIMATES:")
        print(f"Intercept (β₀)            : {stats_result['beta_0']:>10.2f} (SE = {stats_result['se_beta_0']:.2f})")
        print(f"                            95% CI: [{stats_result['ci_lower_beta_0']:.2f}, {stats_result['ci_upper_beta_0']:.2f}]")
        print(f"                            t = {stats_result['t_stat_beta_0']:.3f}, p = {stats_result['p_value_beta_0']:.4f}")
        print()
        
        print(f"Slope (β₁)                : {stats_result['beta_1']:>10.2f} (SE = {stats_result['se_beta_1']:.2f})")
        print(f"                            95% CI: [{stats_result['ci_lower_beta_1']:.2f}, {stats_result['ci_upper_beta_1']:.2f}]")
        print(f"                            t = {stats_result['t_stat_beta_1']:.3f}, p = {stats_result['p_value_beta_1']:.4f}")
        print()
        
        # Interpretation of slope
        if stats_result['beta_1'] > 0:
            print(f"INTERPRETATION: For each 1-unit increase in supernormal stimuli score,")
            print(f"                revenue increases by ${stats_result['beta_1']:,.0f} on average.")
        else:
            print(f"INTERPRETATION: For each 1-unit increase in supernormal stimuli score,")
            print(f"                revenue decreases by ${abs(stats_result['beta_1']):,.0f} on average.")
        print()
        
        print("MODEL FIT STATISTICS:")
        print(f"R²                        : {stats_result['r2']:>10.4f}")
        print(f"ΔR² (vs null model)       : {stats_result['delta_r2']:>10.4f}")
        print(f"Correlation (r)           : {stats_result['correlation']:>10.4f}")
        print(f"RMSE                      : {stats_result['rmse']:>10.2f}")
        print()
        
        print("MODEL SIGNIFICANCE:")
        print(f"F-statistic               : {stats_result['f_statistic']:>10.3f}")
        print(f"F p-value                 : {stats_result['f_p_value']:>10.4f}")
        print()
        
        # Statistical significance interpretation
        alpha = 0.05
        if stats_result['p_value_beta_1'] < alpha:
            print("CONCLUSION: ✓ STATISTICALLY SIGNIFICANT relationship found")
            print(f"            The supernormal stimuli score is a significant predictor of revenue")
            print(f"            (p = {stats_result['p_value_beta_1']:.4f} < α = {alpha})")
            
            if stats_result['beta_1'] > 0:
                print("            → Higher supernormal stimuli scores INCREASE revenue")
            else:
                print("            → Higher supernormal stimuli scores DECREASE revenue")
        else:
            print("CONCLUSION: ✗ NO SIGNIFICANT relationship found")
            print(f"            The supernormal stimuli score is not a significant predictor of revenue")
            print(f"            (p = {stats_result['p_value_beta_1']:.4f} ≥ α = {alpha})")
        
        print("\n" + "=" * 60)
    
    return results

def create_enhanced_visualizations(df, results, dataset_name):
    """Create enhanced visualizations with statistical annotations"""
    figs = {}
    
    # Enhanced regression plot with statistics
    if 'clip_stimuli_index' in df.columns:
        fig_reg, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(df['clip_stimuli_index'], df['revenue'], alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        
        # Regression line
        X = df[['clip_stimuli_index']].values
        y = df['revenue'].values
        reg = LinearRegression().fit(X, y)
        x_line = np.linspace(df['clip_stimuli_index'].min(), df['clip_stimuli_index'].max(), 100)
        y_line = reg.predict(x_line.reshape(-1, 1))
        ax.plot(x_line, y_line, 'r-', alpha=0.8, linewidth=3, label='Regression Line')
        
        # Add confidence interval (approximate)
        stats_result = results.get('clip_stimuli_index', {})
        if stats_result:
            # Calculate prediction intervals (simplified)
            residuals = y - reg.predict(X)
            mse = np.mean(residuals**2)
            std_error = np.sqrt(mse)
            
            y_upper = y_line + 1.96 * std_error
            y_lower = y_line - 1.96 * std_error
            ax.fill_between(x_line, y_lower, y_upper, alpha=0.2, color='red', label='95% Prediction Interval')
        
        ax.set_xlabel('CLIP Supernormal Stimuli Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Revenue (Units Sold)', fontsize=12, fontweight='bold')
        ax.set_title(f'{dataset_name} - Supernormal Stimuli vs Revenue', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add statistics text box
        if stats_result:
            stats_text = (f"β₁ = {stats_result['beta_1']:.2f}\n"
                         f"95% CI: [{stats_result['ci_lower_beta_1']:.2f}, {stats_result['ci_upper_beta_1']:.2f}]\n"
                         f"R² = {stats_result['r2']:.3f}\n"
                         f"p = {stats_result['p_value_beta_1']:.4f}")
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                   verticalalignment='top', fontsize=10, fontfamily='monospace')
        
        fig_reg.tight_layout()
        figs['regression'] = fig_reg
    
    # Individual feature correlations bar chart
    if 'clip_muscular_male' in df.columns:
        fig_corr, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate correlations for individual features
        features = ['clip_muscular_male', 'clip_bright_colors', 'clip_scary_villain']
        feature_names = ['Muscular\nMale', 'Bright\nColors', 'Scary\nVillain']
        correlations = []
        
        for feat in features:
            if feat in df.columns:
                correlation = np.corrcoef(df[feat], df['revenue'])[0, 1]
                correlations.append(correlation)
            else:
                correlations.append(0)
        
        # Create bar chart
        bars = ax.bar(feature_names, correlations, color=['#1f77b4', '#2ca02c', '#d62728'], 
                     alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_ylabel('Correlation with Revenue', fontsize=12, fontweight='bold')
        ax.set_title(f'{dataset_name} - Individual Feature Correlations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
        
        # Add correlation values on top of bars
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., 
                   height + (0.02 if height >= 0 else -0.05),
                   f'{corr:.3f}', ha='center', 
                   va='bottom' if height >= 0 else 'top',
                   fontsize=11, fontweight='bold')
        
        # Set y-axis limits with some padding
        y_min = min(correlations) - 0.1 if correlations else -0.1
        y_max = max(correlations) + 0.1 if correlations else 0.1
        ax.set_ylim(y_min, y_max)
        
        fig_corr.tight_layout()
        figs['correlations'] = fig_corr
    
    # Summary statistics table as a figure
    if results:
        fig_table, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        headers = ['Model', 'β₁ (Slope)', '95% CI', 'p-value', 'R²', 'Significance']
        
        for var_name, stats_result in results.items():
            significance = "✓ Significant" if stats_result['p_value_beta_1'] < 0.05 else "✗ Not Significant"
            ci_text = f"[{stats_result['ci_lower_beta_1']:.2f}, {stats_result['ci_upper_beta_1']:.2f}]"
            
            table_data.append([
                stats_result['model_name'],
                f"{stats_result['beta_1']:.3f}",
                ci_text,
                f"{stats_result['p_value_beta_1']:.4f}",
                f"{stats_result['r2']:.4f}",
                significance
            ])
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Style the table - FIXED: Use proper indexing
        num_cols = len(headers)
        for i in range(num_cols):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color code significance - FIXED: Use proper indexing
        for i, row in enumerate(table_data, 1):  # Start from row 1 (after headers)
            significance_col = num_cols - 1  # Last column index (5 for 6 columns)
            if "✓" in row[significance_col]:
                table[(i, significance_col)].set_facecolor('#E8F5E8')
            else:
                table[(i, significance_col)].set_facecolor('#FFE8E8')
        
        ax.set_title(f'{dataset_name} - Statistical Summary', fontsize=14, fontweight='bold', pad=20)
        fig_table.tight_layout()
        figs['statistics_table'] = fig_table
    
    return figs

def main():
    parser = argparse.ArgumentParser(description='Comprehensive statistical analysis of supernormal stimuli and comic revenue')
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
    
    print("🎨 COMPREHENSIVE SUPERNORMAL STIMULI REVENUE ANALYSIS")
    print("=" * 60)
    print("Research Hypothesis: Higher supernormal stimuli scores → Higher revenue")
    print("=" * 60)
    
    # Load CLIP model
    print("Loading CLIP model...")
    model, preprocess, tokenizer = load_clip_model(args.device)
    print(f"Using device: {args.device}")
    
    all_results = {}
    datasets = []
    
    # Process each dataset
    for dataset_info in [
        ('jan2020', args.jan2020_folder, args.jan2020_csv, 'Jan 2020'),
        ('mar2020', args.mar2020_folder, args.mar2020_csv, 'Mar 2020'),
        ('oct2015', args.oct2015_folder, args.oct2015_csv, 'Oct 2015')
    ]:
        dataset_key, folder, csv_file, display_name = dataset_info
        
        if Path(folder).exists() and Path(csv_file).exists():
            print(f"\n📊 PROCESSING {display_name.upper()} DATA")
            print("-" * 50)
            
            # Load and process
            icv2_data = load_icv2_data(csv_file)
            print(f"Loaded {len(icv2_data)} comics from {display_name} ICV2 data")
            
            gpt_key = None if args.skip_gpt else args.gpt_api_key
            analysis_data = process_image_folder(
                folder, model, preprocess, tokenizer, args.device, gpt_key,
                min_rank=args.min_rank, max_rank=args.max_rank
            )
            
            merged_data = merge_with_icv2_data(analysis_data, icv2_data)
            print(f"Successfully matched {len(merged_data)} comics with revenue data")
            
            if len(merged_data) > 0:
                # Run comprehensive analysis
                dataset_results = run_comprehensive_regression_analysis(merged_data, display_name)
                all_results[dataset_key] = dataset_results
                
                # Create enhanced visualizations
                figs = create_enhanced_visualizations(merged_data, dataset_results, display_name)
                if 'regression' in figs:
                    figs['regression'].savefig(f'{dataset_key}_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
                if 'statistics_table' in figs:
                    figs['statistics_table'].savefig(f'{dataset_key}_statistics_table.png', dpi=300, bbox_inches='tight')
                
                # Save detailed results
                merged_data.to_csv(f'{dataset_key}_detailed_results.csv', index=False)
                
                # Add to combined dataset
                merged_data['dataset'] = display_name
                datasets.append(merged_data)
    
    # Combined analysis
    if len(datasets) >= 2:
        print("\n📊 COMBINED ANALYSIS ACROSS ALL DATASETS")
        print("-" * 50)
        
        combined_df = pd.concat(datasets, ignore_index=True)
        print(f"Combined dataset: {len(combined_df)} comics across {len(datasets)} time periods")
        
        # Run comprehensive combined analysis
        combined_results = run_comprehensive_regression_analysis(combined_df, "Combined Analysis")
        
        # Create enhanced combined visualizations
        figs = create_enhanced_visualizations(combined_df, combined_results, "Combined Analysis")
        if 'regression' in figs:
            figs['regression'].savefig('combined_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        if 'statistics_table' in figs:
            figs['statistics_table'].savefig('combined_statistics_table.png', dpi=300, bbox_inches='tight')
        
        # Save combined results
        combined_df.to_csv('combined_comprehensive_results.csv', index=False)
        
        # Final summary
        print("\n" + "="*80)
        print("FINAL RESEARCH CONCLUSION")
        print("="*80)
        
        clip_stats = combined_results.get('clip_stimuli_index', {})
        if clip_stats:
            if clip_stats['p_value_beta_1'] < 0.05:
                direction = "INCREASE" if clip_stats['beta_1'] > 0 else "DECREASE"
                print(f"✓ SIGNIFICANT EVIDENCE FOUND:")
                print(f"  Higher supernormal stimuli scores significantly {direction} comic revenue")
                print(f"  β₁ = {clip_stats['beta_1']:.3f} (95% CI: [{clip_stats['ci_lower_beta_1']:.3f}, {clip_stats['ci_upper_beta_1']:.3f}])")
                print(f"  p = {clip_stats['p_value_beta_1']:.4f}, R² = {clip_stats['r2']:.4f}")
                print(f"  Effect size: Each 1-unit increase in stimuli score → ${clip_stats['beta_1']:,.0f} revenue change")
            else:
                print(f"✗ NO SIGNIFICANT EVIDENCE FOUND:")
                print(f"  Supernormal stimuli scores do not significantly predict comic revenue")
                print(f"  p = {clip_stats['p_value_beta_1']:.4f} ≥ 0.05")
                print(f"  The relationship could be due to random chance")
    
    plt.show()
    print("\n✅ Comprehensive analysis complete!")
    print("📄 Check the generated CSV files and PNG visualizations for detailed results.")

if __name__ == "__main__":
    main()