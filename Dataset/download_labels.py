from huggingface_hub import hf_hub_download

REPO_ID = "nkp37/OpenVid-1M"

csv_path = hf_hub_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    filename="data/train/OpenVidHD.csv",   # or data/train/OpenVid-1M.csv
    local_dir="openvid_metadata",
    local_dir_use_symlinks=False,
)

print("Downloaded:", csv_path)