import os
import requests
from tqdm import tqdm
import subprocess
import sys


def download_qvim_dev_dataset(data_dir: str = "data"):
    URL = "https://cloud.cp.jku.at/index.php/s/R2tJQnPmxp6RSSz/download/qvim-dev.zip"
    zip_file = os.path.join(data_dir, "qvim-dev.zip")

    if os.path.exists(zip_file):
        print(f"{zip_file} already exists. Skipping download. {URL}")
    else:
        download_zip(URL, zip_file)

    if os.path.exists(os.path.join(data_dir, "qvim-dev")):
        print(f"qvim-dev already exists. Skipping extraction.")
    else:
        extract_zip(zip_file, os.path.join(data_dir, 'qvim-dev'))


def download_vimsketch_dataset(data_dir: str = "data", source: str = "gdrive"):


    # if os.path.exists(zip_file):
    #     print(f"{zip_file} already exists. Skipping download. {URL}")
    # else:
    #     download_zip(URL, zip_file)

    # if os.path.exists(os.path.join(data_dir, "Vim_Sketch_Dataset")):
    #     print(f"Vim_Sketch_Dataset already exists. Skipping extraction.")
    # else:
    #     extract_zip(zip_file, data_dir)

    if source == "zonedo":
        URL = "https://zenodo.org/records/2596911/files/Vim_Sketch_Dataset.zip?download=1"
        zip_file = os.path.join(data_dir, "VimSketch.zip")

        if os.path.exists(os.path.join(data_dir, "Vim_Sketch_Dataset")):
            print(f"Vim_Sketch_Dataset already exists. Skipping everything.")
        elif os.path.exists(zip_file):
            print(f"{zip_file} already exists. Skipping download. {URL}")
            extract_zip(zip_file, data_dir)
        else:
            download_zip(URL, zip_file)
            extract_zip(zip_file, data_dir)

    elif source == "gdrive": # Zonedo zip file was corrupted, so I placed a tar file on gdrive
        import tarfile
        
        GDRIVE_ID = "1qc8khcH0ipm2YBaUXXAk0qJR8TWN8k1k"
        tar_file = os.path.join(data_dir, "VimSketch.tar")
        dataset_path = os.path.join(data_dir, "Vim_Sketch_Dataset")
        
        # Skip if dataset already exists
        if os.path.exists(dataset_path):
            print(f"Dataset directory ready at {dataset_path}")
            return
        else:
            print(f"Dataset directory not found at {dataset_path}.")
        
        # Make sure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Download tar file if it doesn't exist
        if not os.path.exists(tar_file):
            print(f"Downloading VimSketch.tar from Google Drive...")
            subprocess.run([
                "gdown", 
                "--id", GDRIVE_ID, 
                "--output", tar_file
            ], check=True)
        
        # Extract tar file
        print(f"Extracting {tar_file}...")
        with tarfile.open(tar_file) as tar:
            tar.extractall(path=data_dir)
        
        print(f"Dataset directory ready at {dataset_path}")

def download_qvim_eval_dataset(data_dir: str = "data"):
    #
    URL = "https://zenodo.org/records/2596911/files/Vim_Sketch_Dataset.zip?download=1"
    zip_file = os.path.join(data_dir, "VimSketch.zip")

    if os.path.exists(zip_file):
        print(f"{zip_file} already exists. Skipping download. {URL}")
    else:
        download_zip(URL, zip_file)

    if os.path.exists(os.path.join(data_dir, "Vim_Sketch_Dataset")):
        print(f"Vim_Sketch_Dataset already exists. Skipping extraction.")
    else:
        extract_zip(zip_file, data_dir)


def download_zip(url: str, zip_file: str):

    response = requests.get(url, stream=True)

    if response.status_code != 200:
        raise Exception(f"Failed to download {url}. Status code: {response.status_code}")

    total_size = int(response.headers.get("content-length", 0))  # Get file size in bytes
    block_size = 8192  # Size of each chunk

    with open(zip_file, "wb") as file, tqdm(
            total=total_size, unit="B", unit_scale=True, desc=f"Downloading {zip_file}"
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))

    print(f"Download completed: {zip_file}")


def extract_zip(zip_file: str, extract_to_dir: str):
    """Extracts a ZIP file using 7zip from Conda."""

    try:
        if sys.platform == "darwin":
            # absolute path to make sure we're using system unzip, not conda
            subprocess.run(["/usr/bin/unzip", zip_file, "-d", extract_to_dir], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        else:
            subprocess.run(["7z", "x", zip_file, f"-o{extract_to_dir}"])
    except subprocess.CalledProcessError as e:
        print(f"Error extracting {zip_file}: {e}")
