import csv
import requests
import re
import time
import os
from pathlib import Path
import json

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

def clean_title_for_search(title):
    """Clean comic title for searching"""
    # Remove issue number, price, and special characters
    title = re.sub(r'#\d+', '', title)
    title = re.sub(r'\$[\d.]+', '', title)
    title = re.sub(r'[^\w\s]', '', title)
    title = title.strip()
    return title

def search_comic_vine_api(title, session):
    """Search Comic Vine API (free alternative)"""
    # Comic Vine API - you need to get a free API key from https://comicvine.gamespot.com/api/
    api_key = os.getenv("COMIC_VINE_API_KEY", "")
    
    if api_key == "":
        print("⚠️  Please get a Comic Vine API key from https://comicvine.gamespot.com/api/")
        print("⚠️  Set it as an environment variable: COMIC_VINE_API_KEY")
        return None
    
    url = "https://comicvine.gamespot.com/api/search/"
    params = {
        'api_key': api_key,
        'format': 'json',
        'query': title,
        'resources': 'issue',
        'limit': 10
    }
    
    try:
        response = session.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        print(f"  🔍 API Response status: {data.get('status_code', 'unknown')}")
        print(f"  📊 Results found: {len(data.get('results', []))}")
        
        if data.get('error') == 'OK' and data.get('results'):
            for i, issue in enumerate(data['results']):
                print(f"    [{i+1}] {issue.get('name', 'Unknown')} - {issue.get('volume', {}).get('name', 'Unknown Volume')}")
                
                # Check if it's a Marvel comic (be more flexible)
                publisher_name = issue.get('volume', {}).get('publisher', {}).get('name', '').lower()
                print(f"        Publisher: {publisher_name}")
                
                if 'marvel' in publisher_name:
                    if issue.get('image') and issue['image'].get('medium_url'):
                        print(f"        ✅ Found Marvel cover: {issue['image']['medium_url']}")
                        return issue['image']['medium_url']
                
                # Also try with any image if Marvel search fails
                if i == 0 and issue.get('image') and issue['image'].get('medium_url'):
                    print(f"        📸 Fallback image available: {issue['image']['medium_url']}")
        
        else:
            print(f"  ❌ API Error: {data.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"  ❌ Error searching Comic Vine for {title}: {e}")
    
    return None

def download_from_google_images(title, output_dir, session):
    """Alternative: Use Google Images search (educational purposes)"""
    search_query = f"{title} comic cover marvel"
    
    # This is a simplified approach - in practice you'd need to handle Google's anti-bot measures
    # For educational purposes, we'll create placeholder images
    
    # Create a placeholder image filename
    clean_title = re.sub(r'[^\w\s-]', '', title)
    clean_title = re.sub(r'\s+', '_', clean_title.strip()).lower()
    
    placeholder_file = output_dir / f"{clean_title}_placeholder.txt"
    
    with open(placeholder_file, 'w') as f:
        f.write(f"Search query: {search_query}\n")
        f.write(f"Original title: {title}\n")
        f.write("This is a placeholder - replace with actual image scraping logic\n")
    
    return str(placeholder_file)

def create_manual_download_list(marvel_comics, output_file):
    """Create a list of comics for manual download"""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'title', 'search_query', 'google_images_url', 'notes'])
        
        for comic in marvel_comics:
            clean_title = clean_title_for_search(comic['title'])
            search_query = f"{clean_title} comic cover marvel"
            google_url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}&tbm=isch"
            
            writer.writerow([
                comic['rank'],
                comic['title'],
                search_query,
                google_url,
                "Click link to manually download cover"
            ])

def download_image(img_url, filename, session):
    """Download image from URL"""
    try:
        response = session.get(img_url, timeout=15)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            f.write(response.content)
        return True
        
    except Exception as e:
        print(f"    ❌ Error downloading image {img_url}: {e}")
        return False

def main():
    csv_file = "SNS_Comics - Sheet1.csv"
    output_dir = Path("marvel_covers")
    output_dir.mkdir(exist_ok=True)
    
    # Read Marvel comics from CSV
    print("Reading Marvel comics from CSV...")
    marvel_comics = read_marvel_comics(csv_file)
    print(f"Found {len(marvel_comics)} Marvel comics")
    
    # Create session for connection pooling
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    print("\n🎯 Alternative Methods to Get Marvel Covers:")
    print("1. Comic Vine API (recommended)")
    print("2. Manual download list")
    print("3. Placeholder files for testing")
    
    choice = input("\nChoose method (1/2/3): ").strip()
    
    if choice == "1":
        print("\n📝 Using Comic Vine API...")
        successful_downloads = 0
        
        for i, comic in enumerate(marvel_comics[:5]):  # Test with first 5
            print(f"\n[{i+1}/5] Processing: {comic['title']}")
            
            clean_title = clean_title_for_search(comic['title'])
            print(f"  🔍 Searching for: '{clean_title}'")
            
            img_url = search_comic_vine_api(clean_title, session)
            
            if img_url:
                print(f"  ✅ Found image: {img_url}")
                
                # Download the image
                filename = output_dir / f"{comic['rank']}_{clean_title.replace(' ', '_').lower()}.jpg"
                if download_image(img_url, filename, session):
                    print(f"  📥 Downloaded: {filename}")
                    successful_downloads += 1
                else:
                    print(f"  ❌ Failed to download image")
            else:
                print(f"  ❌ No image found for: {comic['title']}")
            
            time.sleep(1)  # Be nice to the API
        
        print(f"\n🎉 Successfully downloaded {successful_downloads} covers using Comic Vine API")
    
    elif choice == "2":
        print("\n📋 Creating manual download list...")
        create_manual_download_list(marvel_comics, "marvel_manual_download_list.csv")
        print("✅ Created 'marvel_manual_download_list.csv'")
        print("📝 Open this file to get Google Images links for manual download")
    
    elif choice == "3":
        print("\n🔧 Creating placeholder files for testing...")
        for i, comic in enumerate(marvel_comics[:10]):
            placeholder_file = download_from_google_images(comic['title'], output_dir, session)
            print(f"✅ Created placeholder: {placeholder_file}")
        
        print(f"\n🎉 Created {min(10, len(marvel_comics))} placeholder files")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
