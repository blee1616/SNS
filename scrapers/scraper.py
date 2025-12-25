import csv
import requests
from bs4 import BeautifulSoup
import re
import time
import os
from pathlib import Path
from urllib.parse import urljoin

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

def clean_title_for_url(title):
    """Convert comic title to Marvel URL format: title_2015_issue_number"""
    # Extract issue number first
    issue_match = re.search(r'#(\d+)', title)
    issue_number = issue_match.group(1) if issue_match else '1'
    
    # Remove issue number and price from title
    title_clean = re.sub(r'#\d+', '', title)
    title_clean = re.sub(r'\$[\d.]+', '', title_clean)
    
    # Remove special characters and replace spaces with underscores
    title_clean = re.sub(r'[^\w\s-]', '', title_clean)
    title_clean = re.sub(r'\s+', '_', title_clean.strip())
    title_clean = title_clean.lower()
    
    # Format as title_2015_issue_number (assuming 2015 based on your dataset)
    url_title = f"{title_clean}_2015_{issue_number}"
    
    return url_title

def search_marvel_comic(title, session):
    """Search for comic on Marvel website and find the correct page"""
    # Extract issue number and clean title
    issue_match = re.search(r'#(\d+)', title)
    issue_number = issue_match.group(1) if issue_match else '1'
    
    # Clean title for search
    clean_title = re.sub(r'#\d+', '', title)
    clean_title = re.sub(r'\$[\d.]+', '', clean_title)
    clean_title = re.sub(r'[^\w\s]', ' ', clean_title)
    clean_title = ' '.join(clean_title.split())  # normalize whitespace
    
    # First, try to construct direct URL based on known pattern
    direct_url = try_direct_marvel_url(title, session)
    if direct_url:
        return direct_url
    
    # Fallback to Google search for Marvel comic
    return search_google_for_marvel_comic(title, session)

def try_direct_marvel_url(title, session):
    """Try to construct Marvel comic URL directly using known patterns"""
    # Extract issue number and clean title
    issue_match = re.search(r'#(\d+)', title)
    issue_number = issue_match.group(1) if issue_match else '1'
    
    # Clean title for URL
    clean_title = re.sub(r'#\d+', '', title)
    clean_title = re.sub(r'\$[\d.]+', '', clean_title)
    clean_title = re.sub(r'[^\w\s-]', '', clean_title)
    clean_title = re.sub(r'\s+', '_', clean_title.strip())
    clean_title = clean_title.lower()
    
    # Known specific URLs for certain comics (based on user knowledge)
    known_urls = {
        "invincible_iron_man_2015_1": 57808,
        # Add more specific IDs as we discover them
    }
    
    url_title = f"{clean_title}_2015_{issue_number}"
    
    # Try known specific ID first
    if url_title in known_urls:
        comic_id = known_urls[url_title]
        direct_url = f"https://www.marvel.com/comics/issue/{comic_id}/{url_title}"
        
        try:
            print(f"  🎯 Trying known URL: {direct_url}")
            response = session.head(direct_url, timeout=5)
            
            if response.status_code == 200:
                print(f"  ✅ Found known URL: {direct_url}")
                return direct_url
                
        except Exception as e:
            print(f"  ❌ Known URL failed: {e}")
    
    # Known Marvel comic ID ranges for 2015 (these are educated guesses)
    # We'll try a range of possible IDs based on common patterns
    base_ids = [57000, 55000, 56000, 58000, 50000, 60000]  # Common ID ranges for 2015
    
    for base_id in base_ids:
        # Try different ID offsets
        for offset in range(0, 2000, 50):  # Check every 50 IDs in range
            comic_id = base_id + offset
            url_title = f"{clean_title}_2015_{issue_number}"
            direct_url = f"https://www.marvel.com/comics/issue/{comic_id}/{url_title}"
            
            try:
                print(f"  � Trying direct URL: {direct_url}")
                response = session.head(direct_url, timeout=5)  # Use HEAD for faster checking
                
                if response.status_code == 200:
                    print(f"  ✅ Found valid URL: {direct_url}")
                    return direct_url
                    
            except Exception as e:
                continue  # Try next ID
                
            # Don't spam the server
            time.sleep(0.1)
    
    return None

def search_google_for_marvel_comic(title, session):
    """Use Google search to find Marvel comic page"""
    # Extract issue number and clean title
    issue_match = re.search(r'#(\d+)', title)
    issue_number = issue_match.group(1) if issue_match else '1'
    
    clean_title = re.sub(r'#\d+', '', title)
    clean_title = re.sub(r'\$[\d.]+', '', clean_title)
    clean_title = re.sub(r'[^\w\s]', ' ', clean_title)
    clean_title = ' '.join(clean_title.split())
    
    # Google search for Marvel comic
    query = f'site:marvel.com/comics/issue "{clean_title}" #{issue_number} 2015'
    google_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    
    try:
        print(f"  🔍 Google search: {query}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = session.get(google_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for Marvel comic links in Google results
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                if 'marvel.com/comics/issue/' in href and 'url?q=' in href:
                    # Extract the actual Marvel URL from Google's redirect
                    start = href.find('url?q=') + 6
                    end = href.find('&', start)
                    if end == -1:
                        end = len(href)
                    
                    marvel_url = href[start:end]
                    if marvel_url.startswith('https://www.marvel.com/comics/issue/'):
                        print(f"  ✅ Found via Google: {marvel_url}")
                        return marvel_url
                        
    except Exception as e:
        print(f"  ❌ Google search error: {e}")
    
    return None



def get_comic_cover_image(comic_url, session):
    """Extract cover image URL from Marvel comic page"""
    try:
        response = session.get(comic_url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Updated selectors based on current Marvel website structure
        cover_selectors = [
            '.BackgroundContainer__Image',  # Main cover image class
            'img.hsDdd',  # Marvel's image class
            'img[alt*="cover"]',
            'img[src*="cdn.marvel.com"]'  # Marvel CDN images
        ]
        
        for selector in cover_selectors:
            img_tag = soup.select_one(selector)
            if img_tag and img_tag.get('src'):
                img_url = img_tag['src']
                # Ensure HTTPS
                if img_url.startswith('//'):
                    img_url = 'https:' + img_url
                elif img_url.startswith('/'):
                    img_url = urljoin(comic_url, img_url)
                
                # If it's a Marvel CDN image, it's likely the cover
                if 'cdn.marvel.com' in img_url:
                    return img_url
        
        # Fallback: get the first image with Marvel CDN URL
        all_images = soup.find_all('img')
        for img in all_images:
            src = img.get('src', '')
            if 'cdn.marvel.com' in src and '/mg/' in src:  # Marvel image path pattern
                if src.startswith('//'):
                    src = 'https:' + src
                elif src.startswith('/'):
                    src = urljoin(comic_url, src)
                return src
                
    except Exception as e:
        print(f"Error getting cover image from {comic_url}: {e}")
    
    return None

def download_image(img_url, filename, session):
    """Download image from URL"""
    try:
        response = session.get(img_url, timeout=15)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            f.write(response.content)
        return True
        
    except Exception as e:
        print(f"Error downloading image {img_url}: {e}")
        return False

def debug_page_structure(comic_url, session):
    """Debug function to see what's on the Marvel page"""
    try:
        response = session.get(comic_url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        print(f"  🔍 Page title: {soup.title.string if soup.title else 'No title'}")
        
        # Look for all images and their attributes
        all_images = soup.find_all('img')
        print(f"  📷 Found {len(all_images)} images on page")
        
        for i, img in enumerate(all_images[:5]):  # Show first 5 images
            src = img.get('src', 'No src')
            alt = img.get('alt', 'No alt')
            class_attr = img.get('class', [])
            print(f"    Image {i+1}: src='{src[:50]}...', alt='{alt[:30]}...', class={class_attr}")
        
        # Look for specific patterns
        patterns_to_check = ['cover', 'hero', 'main', 'comic', 'issue']
        for pattern in patterns_to_check:
            elements = soup.find_all(attrs={'class': re.compile(pattern, re.I)})
            if elements:
                print(f"  🎯 Found {len(elements)} elements with class containing '{pattern}'")
                
    except Exception as e:
        print(f"  ❌ Debug error: {e}")

def debug_search_page(search_url, session):
    """Debug what's actually on Marvel's search page"""
    try:
        response = session.get(search_url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        print(f"    🔍 Page title: {soup.title.string if soup.title else 'No title'}")
        
        # Look for any links
        all_links = soup.find_all('a', href=True)
        print(f"    🔗 Total links found: {len(all_links)}")
        
        # Look for different comic link patterns
        patterns_to_try = [
            r'/comics/issue/\d+/',
            r'/comics/',
            r'/issue/',
            r'comic',
            r'marvel'
        ]
        
        for pattern in patterns_to_try:
            matches = soup.find_all('a', href=re.compile(pattern, re.I))
            if matches:
                print(f"    📋 Pattern '{pattern}': {len(matches)} matches")
                # Show first few matches
                for i, match in enumerate(matches[:3]):
                    href = match.get('href', '')
                    text = match.get_text(strip=True)[:30]
                    print(f"      {i+1}. {href} | {text}")
        
        # Look for any text content
        body_text = soup.get_text()
        if 'No results found' in body_text or 'no results' in body_text.lower():
            print(f"    ❌ Page indicates no search results")
        elif len(body_text) < 100:
            print(f"    ⚠️  Very little content on page")
        else:
            print(f"    ✅ Page has content ({len(body_text)} characters)")
            
    except Exception as e:
        print(f"    ❌ Debug error: {e}")

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
    
    successful_downloads = 0
    results = []
    
    for i, comic in enumerate(marvel_comics[:5]):  # Test with first 5 comics
        print(f"\n[{i+1}/5] Processing: {comic['title']}")
        
        # Try searching for the comic
        comic_url = search_marvel_comic(comic['title'], session)
        
        if not comic_url:
            print(f"❌ Could not find Marvel page for: {comic['title']}")
            results.append({**comic, 'status': 'not_found', 'url': '', 'image_file': ''})
            continue
        
        print(f"✅ Found comic page: {comic_url}")
        
        # Get cover image URL
        img_url = get_comic_cover_image(comic_url, session)
        if not img_url:
            print(f"❌ Could not find cover image for: {comic['title']}")
            results.append({**comic, 'status': 'no_image', 'url': comic_url, 'image_file': ''})
            continue
        
        # Download image
        clean_filename = clean_title_for_url(comic['title'])
        img_filename = output_dir / f"{comic['rank']}_{clean_filename}.jpg"
        if download_image(img_url, img_filename, session):
            print(f"✅ Downloaded: {img_filename}")
            successful_downloads += 1
            results.append({**comic, 'status': 'success', 'url': comic_url, 'image_file': str(img_filename)})
        else:
            print(f"❌ Failed to download: {img_url}")
            results.append({**comic, 'status': 'download_failed', 'url': comic_url, 'image_file': ''})
        
        # Be nice to the server
        time.sleep(2)
    
    # Save results
    with open('marvel_scraping_results.csv', 'w', newline='', encoding='utf-8') as f:
        if results:
            fieldnames = list(results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    print(f"\n🎉 Scraping complete!")
    print(f"Successfully downloaded: {successful_downloads} covers")
    print(f"Results saved to: marvel_scraping_results.csv")
    print(f"Images saved to: {output_dir}")

if __name__ == "__main__":
    main()
