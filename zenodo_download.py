import requests
import os

def download_file(url, filename):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded: {filename}")
    else:
        print(f"Error downloading {filename}. Status code: {response.status_code}")
        print(f"URL attempted: {url}")

def get_file_urls(record_id):
    metadata_url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(metadata_url)
    if response.status_code == 200:
        data = response.json()
        return [(file['links']['self'], file['key']) for file in data.get('files', [])]
    else:
        print(f"Error fetching metadata. Status code: {response.status_code}")
        return []

def main():
    record_id = "10424443"
    
    # Create a directory to store the downloaded files
    directory = f"zenodo_{record_id}"
    os.makedirs(directory, exist_ok=True)
    os.chdir(directory)

    # Get file URLs from the record metadata
    file_urls = get_file_urls(record_id)

    if not file_urls:
        print("No files found or error in fetching file list.")
        return

    # Download files
    for url, filename in file_urls:
        download_file(url, filename)
    
    print(f"Download complete. Files saved in '{directory}' directory.")

if __name__ == "__main__":
    main()