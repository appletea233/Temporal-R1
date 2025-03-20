from huggingface_hub import HfApi
import os

def upload_file_to_huggingface(repo_id, file_path, commit_message="Upload file"):
    """
    Uploads a file to a Hugging Face dataset repository.

    Args:
        repo_id (str): The dataset repository ID (e.g., "your-username/your-dataset").
        file_path (str): Path to the file to upload.
        commit_message (str): Commit message for the upload.
    """
    api = HfApi()

    # Compute the relative path to maintain directory structure
    path_in_repo = os.path.relpath(file_path, start=os.path.dirname(directory_path))

    # Upload file to the dataset repo
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=path_in_repo,  # Store with the same relative path
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message
    )
    print(f"Successfully uploaded {file_path} to {repo_id}/{path_in_repo}")

def upload_files_in_directory(directory_path, repo_id):
    """
    Uploads all files in the specified directory and its subdirectories.

    Args:
        directory_path (str): The path to the directory to search for files.
        repo_id (str): The dataset repository ID.
    """
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            upload_file_to_huggingface(repo_id, file_path)

def upload_zips_in_directory(directory_path, repo_id):
    """
    Uploads all files in the specified directory and its subdirectories.

    Args:
        directory_path (str): The path to the directory to search for files.
        repo_id (str): The dataset repository ID.
    """
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith(".zip"):
                file_path = os.path.join(root, filename)
                upload_file_to_huggingface(repo_id, file_path)

def create_repo_if_not_exists(repo_id):
    """
    Create a Hugging Face repository if it does not exist.

    Args:
        repo_id (str): The dataset repository ID (e.g., "your-username/your-dataset").
    """
    api = HfApi()
    try:
        # Check if the repository exists
        api.repo_info(repo_id)
        print(f"Repository {repo_id} already exists.")
    except Exception as e:
        # If it does not exist, create it
        print(f"Creating repository {repo_id}.")
        api.create_repo(repo_id=repo_id, repo_type="dataset")

# Example Usage
directory_path = ""
repo_id = "appletea2333/temporal_r1"  # Replace with your repo ID

# create_repo_if_not_exists(repo_id)
upload_zips_in_directory(directory_path, repo_id)
