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

def show_url_pattern(title):
    """Show what URL pattern we would search for"""
    # Extract issue number and clean title
    issue_match = re.search(r'#(\d+)', title)
    issue_number = issue_match.group(1) if issue_match else '1'
    
    # Clean title for URL
    clean_title = re.sub(r'#\d+', '', title)
    clean_title = re.sub(r'\$[\d.]+', '', clean_title)
    clean_title = re.sub(r'[^\w\s-]', '', clean_title)
    clean_title = re.sub(r'\s+', '_', clean_title.strip())
    clean_title = clean_title.lower()
    
    url_title = f"{clean_title}_2015_{issue_number}"
    pattern_url = f"https://www.marvel.com/comics/issue/XXXXX/{url_title}"
    
    print(f"  🔍 URL Pattern: {pattern_url}")
    print(f"      (where XXXXX should be a random Marvel comic ID like 57808, 55432, etc.)")
    
    return pattern_url

def main():
    """Main function to show URL patterns for Marvel comics"""
    csv_file = "SNS_Comics - Sheet1.csv"
    
    # Read Marvel comics from CSV
    print("Reading Marvel comics from CSV...")
    marvel_comics = read_marvel_comics(csv_file)
    print(f"Found {len(marvel_comics)} Marvel comics\n")
    
    # Show URL patterns for first 10 comics
    for i, comic in enumerate(marvel_comics[:10], 1):
        title = comic['title']
        print(f"[{i}/10] {title}")
        show_url_pattern(title)
        print()

if __name__ == "__main__":
    main()
