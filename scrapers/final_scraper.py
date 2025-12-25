import csv
import requests
from bs4 import BeautifulSoup
import re
import time
from pathlib import Path
from difflib import SequenceMatcher

def read_marvel_comics(csv_file):
    """Read CSV and filter for Marvel comics (MAR symbol)"""
    marvel_comics = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 4 and row[3].strip() == 'MAR':
                marvel_comics.append({
                    'rank': row[0],
                    'title': row[1],
                    'price': row[2],
                    'publisher': row[3]
                })
    
    return marvel_comics

def clean_title(title):
    """Clean title for comparison"""
    clean = re.sub(r'#\d+|\$[\d.]+', '', title)
    return ' '.join(clean.split()).upper()

def similarity(a, b):
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, a, b).ratio()

def check_comic(comic_id, session):
    """Check a single comic ID"""
    url = f"https://www.marvel.com/comics/issue/{comic_id}/"
    
    try:
        response = session.get(url, timeout=10)
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Get title
        title_elem = soup.select_one('h1')
        if not title_elem:
            return None
        title = title_elem.get_text(strip=True)
        
        # Check for October 2015 in page text
        page_text = soup.get_text()
        if not (('october' in page_text.lower() or 'oct' in page_text.lower()) and '2015' in page_text):
            return None
            
        # Get cover image
        cover_url = None
        for img in soup.find_all('img'):
            src = img.get('src', '')
            if 'cdn.marvel.com' in src and 'portrait' in src:
                if src.startswith('//'):
                    cover_url = 'https:' + src
                else:
                    cover_url = src
                break
        
        return {
            'title': title,
            'url': url,
            'cover_url': cover_url
        }
        
    except Exception as e:
        print(f"Error checking {comic_id}: {e}")
        return None

def main():
    """Main function"""
    print("🔍 Starting focused Marvel comic search...")
    
    # Read our Marvel comics
    marvel_comics = read_marvel_comics("SNS_Comics - Sheet1.csv")
    print(f"📚 Loaded {len(marvel_comics)} Marvel comics from dataset")
    
    # Create session
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    # Check a focused range around known good IDs
    matches = []
    output_dir = Path("marvel_covers_final")
    output_dir.mkdir(exist_ok=True)
    
    # Focus on the range we know has 2015 comics
    start_id = 57000
    end_id = 58500
    
    print(f"🔍 Checking Marvel IDs {start_id} to {end_id}...")
    
    for comic_id in range(start_id, end_id + 1):
        if comic_id % 50 == 0:
            print(f"  📊 Checked up to ID {comic_id}... Found {len(matches)} matches so far")
        
        # Check this comic
        comic_info = check_comic(comic_id, session)
        if not comic_info:
            continue
            
        print(f"  ✅ Found October 2015 comic: {comic_info['title']} (ID: {comic_id})")
        
        # Compare with our dataset
        clean_marvel_title = clean_title(comic_info['title'])
        
        best_match = None
        best_similarity = 0
        
        for dataset_comic in marvel_comics:
            clean_dataset_title = clean_title(dataset_comic['title'])
            sim_score = similarity(clean_marvel_title, clean_dataset_title)
            
            if sim_score > best_similarity:
                best_similarity = sim_score
                best_match = dataset_comic
        
        # If good match found
        if best_similarity >= 0.75:
            print(f"    🎯 MATCH! {best_similarity:.2f} similarity with '{best_match['title']}'")
            
            # Download cover if available
            if comic_info['cover_url']:
                rank_num = int(best_match['rank']) if best_match['rank'].isdigit() else 999
                filename = f"rank_{rank_num:03d}_{comic_id}_{best_match['title'][:30].replace(' ', '_').replace('#', '')}.jpg"
                filename = re.sub(r'[^\w\-_\.]', '', filename)
                filepath = output_dir / filename
                
                try:
                    img_response = session.get(comic_info['cover_url'], timeout=15)
                    if img_response.status_code == 200:
                        with open(filepath, 'wb') as f:
                            f.write(img_response.content)
                        print(f"    📥 Downloaded: {filename}")
                    else:
                        print(f"    ❌ Failed to download cover")
                except Exception as e:
                    print(f"    ❌ Download error: {e}")
            
            matches.append({
                'rank': best_match['rank'],
                'dataset_title': best_match['title'],
                'marvel_title': comic_info['title'],
                'marvel_id': comic_id,
                'similarity': best_similarity,
                'url': comic_info['url'],
                'cover_url': comic_info['cover_url']
            })
        else:
            print(f"    ⚠️  Best match only {best_similarity:.2f} with '{best_match['title'] if best_match else 'None'}'")
        
        # Small delay to be respectful
        time.sleep(0.3)
    
    # Save results
    if matches:
        with open('final_marvel_results.csv', 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['rank', 'dataset_title', 'marvel_title', 'marvel_id', 'url', 'similarity', 'cover_url']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(matches)
    
    print(f"\n🎉 Search complete!")
    print(f"📊 Found {len(matches)} matches")
    print(f"📁 Images saved to: {output_dir}")
    print(f"📄 Results saved to: final_marvel_results.csv")

if __name__ == "__main__":
    main()
