import os
import urllib
import torch
from qvim_mn_baseline.ex_qbv import QVIMModule


device = "cuda" if torch.cuda.is_available() else "cpu"

# GitHub release URL and local directory to store models
model_url = "https://github.com/Jonathan-Greif/QBV/releases/download/v1.0.0/"
model_dir = "resources"

# Pretrained models dictionary
pretrained_models = {
    "model_id": urllib.parse.urljoin(model_url, "ct_nt_xent_fold0mn10d10s32_01.pt")
}


def get_model(name):
    download_model()

    return QVIMModule.load_from_checkpoint(pretrained_models[name])


def download_model(name, url):
    # Define the file path in the local directory
    file_path = os.path.join(model_dir, name + ".pt")
    os.makedirs(model_dir, exist_ok=True)
    if not os.path.exists(file_path):
        print(f"Downloading {name} model...")
        try:
            # Download the file
            urllib.request.urlretrieve(url, file_path)
            print(f"Model {name} downloaded and saved to {file_path}")
        except Exception as e:
            print(f"Failed to download {name}: {e}")

