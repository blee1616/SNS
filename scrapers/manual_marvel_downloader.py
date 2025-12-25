import csv
import requests
from pathlib import Path
import time

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

def create_manual_download_script():
    """Create a comprehensive solution for getting Marvel covers"""
    
    marvel_comics = read_marvel_comics("SNS_Comics - Sheet1.csv")
    
    # Create directories
    Path("marvel_covers").mkdir(exist_ok=True)
    
    # Create a detailed CSV with multiple search approaches
    with open('marvel_cover_sources.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'rank', 'title', 'suggested_filename', 
            'google_images_search', 'duckduckgo_search', 
            'marvel_search', 'comixology_search',
            'notes'
        ])
        
        for comic in marvel_comics:
            # Clean title for searching
            clean_title = comic['title'].replace('#', '').replace('$', '').strip()
            filename = f"{comic['rank']}_{clean_title.lower().replace(' ', '_').replace('-', '_')}.jpg"
            
            # Create multiple search URLs
            google_search = f"https://www.google.com/search?q={clean_title.replace(' ', '+')}+comic+cover+marvel&tbm=isch"
            duck_search = f"https://duckduckgo.com/?q={clean_title.replace(' ', '+')}+comic+cover+marvel&iax=images&ia=images"
            marvel_search = f"https://www.marvel.com/search?query={clean_title.replace(' ', '+')}"
            comixology_search = f"https://www.comixology.com/search?search={clean_title.replace(' ', '+')}"
            
            writer.writerow([
                comic['rank'], 
                comic['title'], 
                filename,
                google_search,
                duck_search,
                marvel_search,
                comixology_search,
                "Multiple sources for manual download"
            ])
    
    print("✅ Created 'marvel_cover_sources.csv' with search links")
    
    # Create an HTML helper file
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Marvel Comic Cover Downloader</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .comic { border: 1px solid #ddd; margin: 10px 0; padding: 15px; }
        .comic h3 { color: #c41e3a; margin-top: 0; }
        .search-links { margin: 10px 0; }
        .search-links a { 
            display: inline-block; margin: 5px 10px 5px 0; 
            padding: 8px 12px; background: #f8f9fa; 
            text-decoration: none; border: 1px solid #ddd;
            border-radius: 4px;
        }
        .search-links a:hover { background: #e9ecef; }
        .filename { 
            background: #f1f3f4; padding: 8px; 
            font-family: monospace; margin: 10px 0;
            border-radius: 4px;
        }
        .instructions { 
            background: #e8f4f8; padding: 15px; 
            margin-bottom: 20px; border-radius: 8px;
        }
        .google { background-color: #4285f4; color: white; }
        .duck { background-color: #de5833; color: white; }
        .marvel { background-color: #c41e3a; color: white; }
        .comix { background-color: #ff6900; color: white; }
    </style>
</head>
<body>
    <h1>🦸‍♂️ Marvel Comic Cover Downloader</h1>
    
    <div class="instructions">
        <h3>📋 How to Download:</h3>
        <ol>
            <li><strong>Click search links</strong> below to find comic covers</li>
            <li><strong>Right-click</strong> on the best quality cover image</li>
            <li><strong>Save As</strong> using the suggested filename</li>
            <li><strong>Save to</strong> your <code>marvel_covers</code> folder</li>
            <li><strong>Look for</strong> high-resolution images (avoid thumbnails)</li>
        </ol>
        <p><strong>💡 Tip:</strong> Google Images usually has the best quality covers!</p>
    </div>
    
"""
    
    for i, comic in enumerate(marvel_comics):
        if i >= 20:  # Limit to first 20 for the HTML file
            break
            
        clean_title = comic['title'].replace('#', '').replace('$', '').strip()
        filename = f"{comic['rank']}_{clean_title.lower().replace(' ', '_').replace('-', '_')}.jpg"
        
        google_search = f"https://www.google.com/search?q={clean_title.replace(' ', '+')}+comic+cover+marvel+2015&tbm=isch"
        duck_search = f"https://duckduckgo.com/?q={clean_title.replace(' ', '+')}+comic+cover+marvel&iax=images&ia=images"
        marvel_search = f"https://www.marvel.com/search?query={clean_title.replace(' ', '+')}"
        comixology_search = f"https://www.comixology.com/search?search={clean_title.replace(' ', '+')}"
        
        html_content += f"""
    <div class="comic">
        <h3>#{comic['rank']} - {comic['title']}</h3>
        <div class="filename">💾 <strong>Save as:</strong> {filename}</div>
        <div class="search-links">
            <a href="{google_search}" target="_blank" class="google">🔍 Google Images</a>
            <a href="{duck_search}" target="_blank" class="duck">🦆 DuckDuckGo</a>
            <a href="{marvel_search}" target="_blank" class="marvel">🦸‍♂️ Marvel.com</a>
            <a href="{comixology_search}" target="_blank" class="comix">📚 ComiXology</a>
        </div>
    </div>
"""
    
    html_content += """
    <div style="margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px;">
        <h3>📄 For All Comics:</h3>
        <p>Check <strong>marvel_cover_sources.csv</strong> for the complete list of all """ + str(len(marvel_comics)) + """ Marvel comics with search links.</p>
    </div>
</body>
</html>
"""
    
    with open("marvel_downloader.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print("✅ Created 'marvel_downloader.html' - open this in your browser")
    print(f"📊 Found {len(marvel_comics)} Marvel comics total")
    print("💡 HTML file shows first 20 comics, CSV has all comics")

if __name__ == "__main__":
    create_manual_download_script()
