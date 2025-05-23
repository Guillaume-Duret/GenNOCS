from huggingface_hub import hf_hub_download, list_repo_files

# Specify the repository
#repo_id = "stabilityai/stable-diffusion-3.5-medium"
repo_id = "stabilityai/stable-diffusion-3.5-large-turbo"

# List all files in the repository
file_names = list_repo_files(repo_id)

# Download all files
for file_name in file_names:
    hf_hub_download(repo_id=repo_id, filename=file_name)
    print(f"Downloaded: {file_name}")

