import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def copy_identity_folder(args):
    source_folder, destination_folder = args
    try:
        shutil.copytree(source_folder, destination_folder, symlinks=False)
        return 1
    except FileExistsError:
        return 0

original_dataset_path = '/home/aryan/FSCIL/datasets/ms1m/'
destination_path = '/home/aryan/FSCIL/datasets/ms1m_small/'
percentage_to_keep = 12

os.makedirs(destination_path, exist_ok=True)

identity_folders = [f for f in os.listdir(original_dataset_path) if os.path.isdir(os.path.join(original_dataset_path, f))]

num_identity_folders_to_keep = int(len(identity_folders) * percentage_to_keep / 100)
selected_identity_folders = random.sample(identity_folders, num_identity_folders_to_keep)

args_list = [(os.path.join(original_dataset_path, identity_folder),
              os.path.join(destination_path, identity_folder))
             for identity_folder in selected_identity_folders]

with ThreadPoolExecutor() as executor, tqdm(total=len(args_list)) as pbar:
    for result in executor.map(copy_identity_folder, args_list):
        pbar.update(result)
        pbar.set_postfix(Completed=f"{pbar.n}/{pbar.total}")

print(f"Reduced the dataset to {percentage_to_keep}% of its original size with identity folder structure preserved.")