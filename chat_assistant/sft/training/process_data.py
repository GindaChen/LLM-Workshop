import datasets
from datasets import load_dataset
from huggingface_hub import HfApi

# Load the original dataset
ds = load_dataset("smangrul/ultrachat-10k-chatml")

# Take a 1k subset of the train dataset
ds_train_1k = ds["train"].shuffle(seed=42).select(range(1000))

# Take a 100 subset of the test dataset
ds_test_100 = ds["test"].shuffle(seed=42).select(range(100))

# Create a new dataset with the 1k train subset and 100 test subset
ds_1k = datasets.DatasetDict({
    "train": ds_train_1k,
    "test": ds_test_100
})

# Push the dataset to your HuggingFace account
api = HfApi()
repo_id = "GindaChen/ultrachat-1k-chatml"


# Create a new repo and push the dataset
api.create_repo(repo_id=repo_id, repo_type="dataset", private=False)
ds_1k.push_to_hub(repo_id)

print("Dataset processed and uploaded successfully!")
