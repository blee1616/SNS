import csv
import requests
from bs4 import BeautifulSoup
import re
import time
from pathlib import Path
from urllib.parse import urljoin
from difflib import SequenceMatcher

def read_marvel_comics(csv_file):
    """Read CSV and filter for Marvel comics (MAR symbol)"""
    marvel_comics = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 4 and row[3].strip() == 'MAR':
                rank = row[0]
                title = row[1]
                price = row[2]
                publisher = row[3]
                units = row[4] if len(row) > 4 else ''
                revenue = row[5] if len(row) > 5 else ''
                
                marvel_comics.append({
                    'rank': rank,
                    'title': title,
                    'price': price,
                    'publisher': publisher,
                    'units': units,
                    'revenue': revenue
                })
    
    return marvel_comics

def clean_title_for_comparison(title):
    """Clean title for comparison - remove issue numbers, prices, extra spaces"""
    # Remove issue numbers and prices
    clean = re.sub(r'#\d+', '', title)
    clean = re.sub(r'\$[\d.]+', '', clean)
    
    # Remove extra whitespace and normalize
    clean = ' '.join(clean.split()).upper()
    
    return clean

def similarity(a, b):
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, a, b).ratio()

def extract_comic_info(comic_id, session):
    """Extract title and publication date from Marvel comic page"""
    url = f"https://www.marvel.com/comics/issue/{comic_id}/"
    
    try:
        print(f"  🔍 Checking ID {comic_id}...")
        response = session.get(url, timeout=10)
        
        if response.status_code != 200:
            return None, None, None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title - look for the main heading
        title = None
        title_selectors = [
            'h1',
            '.masthead__title',
            '.comic-title',
            '[class*="title"]'
        ]
        
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                title = title_elem.get_text(strip=True)
                if title and len(title) > 5:  # Valid title
                    break
        
        # Extract publication date - look for "PUBLISHED:" section
        pub_date = None
        
        # Look for "PUBLISHED:" text and get the next element
        published_text = soup.find(string=re.compile(r'PUBLISHED:'))
        if published_text:
            # Find the parent element and look for date nearby
            parent = published_text.parent
            if parent:
                # Look for date in the same element or next sibling
                date_text = parent.get_text(strip=True)
                date_match = re.search(r'PUBLISHED:\s*(.*)', date_text)
                if date_match:
                    pub_date = date_match.group(1).strip()
        
        # Alternative: look for date patterns in the page
        if not pub_date:
            # Look for "October 2015" or similar patterns
            page_text = soup.get_text()
            date_patterns = [
                r'October\s+\d{1,2},?\s+2015',
                r'Oct\s+\d{1,2},?\s+2015',
                r'PUBLISHED:\s*([^\n]+)',
                r'Release Date:\s*([^\n]+)'
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    pub_date = match.group(1) if pattern.count('(') > 0 else match.group(0)
                    break
        
        # Extract cover image URL
        cover_url = None
        
        # Look for cover image
        cover_selectors = [
            'img[src*="portrait_uncanny"]',
            'img[src*="cdn.marvel.com"]',
            '.comic-cover img',
            '.hero-image img',
            'img[alt*="cover"]'
        ]
        
        for selector in cover_selectors:
            img_elem = soup.select_one(selector)
            if img_elem and img_elem.get('src'):
                src = img_elem['src']
                if 'cdn.marvel.com' in src:
                    # Ensure full URL
                    if src.startswith('//'):
                        cover_url = 'https:' + src
                    elif src.startswith('/'):
                        cover_url = urljoin(url, src)
                    else:
                        cover_url = src
                    break
        
        print(f"    📚 Title: {title}")
        print(f"    📅 Date: {pub_date}")
        print(f"    🖼️  Cover: {cover_url is not None}")
        
        return title, pub_date, cover_url
        
    except Exception as e:
        print(f"    ❌ Error processing ID {comic_id}: {e}")
        return None, None, None

def is_october_2015(date_str):
    """Check if the date string indicates October 2015"""
    if not date_str:
        return False
        
    date_str = date_str.lower()
    return ('october' in date_str or 'oct' in date_str) and '2015' in date_str

def find_matching_comics(marvel_comics, session, start_id=52000, end_id=60000):
    """Iterate through Marvel IDs and find matching comics"""
    
    print(f"🔍 Searching Marvel IDs {start_id} to {end_id} for October 2015 comics...")
    print(f"📋 Looking for {len(marvel_comics)} Marvel comics from dataset\n")
    
    matches = []
    checked_count = 0
    
    for comic_id in range(start_id, end_id + 1):
        checked_count += 1
        
        # Extract info from Marvel page
        marvel_title, pub_date, cover_url = extract_comic_info(comic_id, session)
        
        if not marvel_title or not is_october_2015(pub_date):
            if checked_count % 100 == 0:
                print(f"  📊 Checked {checked_count} IDs so far...")
            continue
        
        print(f"  ✅ Found October 2015 comic: {marvel_title}")
        
        # Clean the Marvel title for comparison
        clean_marvel_title = clean_title_for_comparison(marvel_title)
        
        # Check against our dataset
        for dataset_comic in marvel_comics:
            clean_dataset_title = clean_title_for_comparison(dataset_comic['title'])
            
            # Calculate similarity
            sim_score = similarity(clean_marvel_title, clean_dataset_title)
            
            # If similarity is high enough (80%+), it's likely a match
            if sim_score >= 0.8:
                print(f"    🎯 MATCH FOUND! Similarity: {sim_score:.2f}")
                print(f"      Dataset: {dataset_comic['title']}")
                print(f"      Marvel:  {marvel_title}")
                print(f"      URL:     https://www.marvel.com/comics/issue/{comic_id}/")
                
                matches.append({
                    'dataset_comic': dataset_comic,
                    'marvel_title': marvel_title,
                    'marvel_id': comic_id,
                    'pub_date': pub_date,
                    'cover_url': cover_url,
                    'similarity': sim_score,
                    'marvel_url': f"https://www.marvel.com/comics/issue/{comic_id}/"
                })
                break
        
        # Be respectful to Marvel's servers
        time.sleep(0.5)
    
    print(f"\n🎉 Search complete! Checked {checked_count} IDs, found {len(matches)} matches")
    return matches

def download_image(img_url, filename, session):
    """Download image from URL"""
    try:
        response = session.get(img_url, timeout=15)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            f.write(response.content)
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {img_url}: {e}")
        return False

def main():
    """Main function"""
    csv_file = "SNS_Comics - Sheet1.csv"
    output_dir = Path("marvel_covers_matched")
    output_dir.mkdir(exist_ok=True)
    
    # Read Marvel comics from dataset
    marvel_comics = read_marvel_comics(csv_file)
    
    # Create session
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    # Find matching comics
    # Expand range to cover more 2015 comics
    matches = find_matching_comics(marvel_comics, session, start_id=55000, end_id=58000)
    
    # Download covers for matches
    successful_downloads = 0
    
    for match in matches:
        dataset_comic = match['dataset_comic']
        cover_url = match['cover_url']
        marvel_id = match['marvel_id']
        
        if cover_url:
            filename = f"rank_{dataset_comic['rank']}_{marvel_id}_{dataset_comic['title'].replace(' ', '_').replace('#', '').replace('$', '')[:50]}.jpg"
            filename = re.sub(r'[^\w\-_\.]', '', filename)  # Remove invalid characters
            filepath = output_dir / filename
            
            print(f"\n📥 Downloading: {filename}")
            if download_image(cover_url, filepath, session):
                print(f"✅ Downloaded: {filepath}")
                successful_downloads += 1
            else:
                print(f"❌ Failed to download")
        
        time.sleep(1)
    
    # Save results
    if matches:
        with open('marvel_matches_results.csv', 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['rank', 'dataset_title', 'marvel_title', 'marvel_id', 'marvel_url', 'pub_date', 'similarity', 'cover_url']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for match in matches:
                writer.writerow({
                    'rank': match['dataset_comic']['rank'],
                    'dataset_title': match['dataset_comic']['title'],
                    'marvel_title': match['marvel_title'],
                    'marvel_id': match['marvel_id'],
                    'marvel_url': match['marvel_url'],
                    'pub_date': match['pub_date'],
                    'similarity': f"{match['similarity']:.3f}",
                    'cover_url': match['cover_url']
                })
    
    print(f"\n🎉 Process complete!")
    print(f"Found {len(matches)} matches")
    print(f"Downloaded {successful_downloads} covers")
    print(f"Results saved to: marvel_matches_results.csv")
    print(f"Images saved to: {output_dir}")

if __name__ == "__main__":
    main()
