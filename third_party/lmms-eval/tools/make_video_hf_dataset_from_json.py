import json
import os
import numpy as np

import pandas as pd
from datasets import Dataset, Features, Image, Sequence, Value
from tqdm import tqdm

# Define the features for the dataset
features = Features(
    {
        "video": Value(dtype="string"),
        "caption": Value(dtype="string"),
        "timestamp": Sequence(Value(dtype="float16")),  # Use Sequence for lists
    }
)

df_items = {
    "video": [],
    "caption": [],
    "timestamp": [],
}

# Load json file
json_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/lihongyu/projects/video_llm/codes/EasyR1/data/tvg_jsons_processed/anet_val_hf.json"
with open(json_path, "r") as f:
    data = json.load(f)


# Iterate over the rows of the data
for cur_meta in data:
    video = cur_meta["video"]
    caption = cur_meta["caption"]
    timestamp = cur_meta["timestamp"]

    use_type = np.float16
    timestamp = [use_type(timestamp[0]), use_type(timestamp[1])]
    # import pdb;pdb.set_trace()
    df_items["video"].append(video)
    df_items["caption"].append(caption)
    df_items["timestamp"].append(timestamp)

import pdb

pdb.set_trace()

df_items = pd.DataFrame(df_items)

dataset = Dataset.from_pandas(df_items, features=features)

hub_dataset_path = "appletea2333/activity_tvg"
dataset.push_to_hub(repo_id=hub_dataset_path, split="test")

# # upload the *zip to huggingface
# from huggingface_hub import HfApi

# def upload_zip_to_huggingface(repo_id, zip_path, commit_message="Upload ZIP file"):
#     """
#     Uploads a ZIP file to a Hugging Face dataset repository.

#     Args:
#         repo_id (str): The dataset repository ID (e.g., "your-username/your-dataset").
#         zip_path (str): Path to the ZIP file to upload.
#         commit_message (str): Commit message for the upload.
#     """
#     api = HfApi()

#     # Upload file to the dataset repo
#     api.upload_file(
#         path_or_fileobj=zip_path,
#         path_in_repo=zip_path.split("/")[-1],  # Store with the same filename
#         repo_id=repo_id,
#         repo_type="dataset",
#         commit_message=commit_message
#     )
#     print(f"Successfully uploaded {zip_path} to {repo_id}")

# # Example Usage for upload all zip in directory
# import os
# directory_path = "/home/tiger/split_zips"
# # Iterate over all files in the directory
# for filename in os.listdir(directory_path):
#     if filename.endswith(".zip"):
#         file_path = os.path.join(directory_path, filename)
#         upload_zip_to_huggingface("lmms-lab/charades_sta", file_path)
