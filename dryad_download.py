import requests
from bs4 import BeautifulSoup
import os

# URL of the Dryad dataset page
dryad_url = "https://datadryad.org/stash/dataset/doi:10.5061/dryad.905qftttm"

# Directory to save the downloaded files
download_dir = "dryad_data"
os.makedirs(download_dir, exist_ok=True)

def get_file_links(dryad_url):
    response = requests.get(dryad_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all links to data files
    file_links = []
    for a_tag in soup.find_all('a', class_='btn btn-default btn-xs'):
        href = a_tag.get('href')
        if href and 'download' in href:
            file_links.append(href)
    
    return file_links

def download_files(file_links, download_dir):
    for link in file_links:
        file_name = os.path.join(download_dir, link.split('/')[-1])
        response = requests.get(link, stream=True)
        
        with open(file_name, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded: {file_name}")

if __name__ == "__main__":
    file_links = get_file_links(dryad_url)
    if file_links:
        download_files(file_links, download_dir)
        print("All files downloaded successfully.")
    else:
        print("No files found for download.")
