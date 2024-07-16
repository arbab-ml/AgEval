import requests
from bs4 import BeautifulSoup
import os

def get_dryad_files(base_url):
    response = requests.get(base_url)
    response.raise_for_status()  # Check if the request was successful
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all file download links
    file_links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if 'stash/downloads/file_stream' in href:
            # Construct the full URL
            full_url = requests.compat.urljoin(base_url, href)
            file_links.append(full_url)
    
    return file_links

def download_file(url, output_path):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()  # Check if the request was successful

    with open(output_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded {output_path}")

# Dryad dataset URL
dataset_url = "https://datadryad.org/stash/dataset/doi:10.5061/dryad.905qftttm"

# Get the file links
file_links = get_dryad_files(dataset_url)

# Create a directory to store the downloaded files
os.makedirs("dryad_files", exist_ok=True)

# Download each file
for i, file_link in enumerate(file_links):
    file_name = f"dryad_files/dryad_file_{i + 1}.zip"
    download_file(file_link, file_name)
