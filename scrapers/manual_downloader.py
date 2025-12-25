import csv
import webbrowser
from pathlib import Path
import re

def read_marvel_comics(csv_file):
    """Read CSV and filter for Marvel comics (MAR symbol)"""
    marvel_comics = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 4 and row[3].strip() == 'MAR':
                rank = row[0]
                title = row[1]
                marvel_comics.append({
                    'rank': rank,
                    'title': title
                })
    
    return marvel_comics

def create_download_helper():
    """Create an HTML file to help with manual downloads"""
    marvel_comics = read_marvel_comics("SNS_Comics - Sheet1.csv")
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Marvel Comic Cover Downloader</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .comic { border: 1px solid #ccc; margin: 10px; padding: 15px; }
        .comic h3 { color: #c41e3a; }
        .links a { margin-right: 10px; padding: 5px 10px; background: #f0f0f0; text-decoration: none; }
        .links a:hover { background: #ddd; }
        .instructions { background: #e8f4f8; padding: 15px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <h1>Marvel Comic Cover Downloader</h1>
    
    <div class="instructions">
        <h3>📋 Instructions:</h3>
        <ol>
            <li>Click the search links below to find comic covers</li>
            <li>Right-click on the cover image and "Save As"</li>
            <li>Name the file: rank_title.jpg (e.g., "1_invincible_iron_man.jpg")</li>
            <li>Save to your marvel_covers folder</li>
        </ol>
    </div>
    
"""
    
    for comic in marvel_comics[:20]:  # First 20 comics
        clean_title = re.sub(r'[^\w\s]', '', comic['title']).strip()
        search_query = f"{clean_title} comic cover".replace(' ', '+')
        
        google_images = f"https://www.google.com/search?q={search_query}&tbm=isch"
        bing_images = f"https://www.bing.com/images/search?q={search_query}"
        
        html_content += f"""
    <div class="comic">
        <h3>#{comic['rank']} - {comic['title']}</h3>
        <div class="links">
            <a href="{google_images}" target="_blank">🔍 Google Images</a>
            <a href="{bing_images}" target="_blank">🔍 Bing Images</a>
        </div>
        <p><strong>Suggested filename:</strong> {comic['rank']}_{clean_title.lower().replace(' ', '_')}.jpg</p>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    with open("marvel_cover_downloader.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print("✅ Created 'marvel_cover_downloader.html'")
    print("📂 Open this file in your browser to start downloading covers manually")
    
    # Create the output directory
    Path("marvel_covers").mkdir(exist_ok=True)
    
    # Open the HTML file automatically
    webbrowser.open("marvel_cover_downloader.html")

if __name__ == "__main__":
    create_download_helper()
