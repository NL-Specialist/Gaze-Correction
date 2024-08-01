import os
import shutil
import random

# Define paths
source_folder = 'Duncan/at_camera'
destination_folder = 'Duncan/at_camera_split'
subfolders = ['train', 'test', 'validate']

# Create destination subfolders if they do not exist
for subfolder in subfolders:
    os.makedirs(os.path.join(destination_folder, subfolder), exist_ok=True)

# List all subfolders in the source folder
image_folders = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))]

# Shuffle image folders to randomize the split
random.shuffle(image_folders)

# Calculate split indices
total_folders = len(image_folders)
train_split = int(0.7 * total_folders)
test_split = int(0.2 * total_folders)
validate_split = total_folders - train_split - test_split

# Split folders
train_folders = image_folders[:train_split]
test_folders = image_folders[train_split:train_split + test_split]
validate_folders = image_folders[train_split + test_split:]

# Function to move folders
def move_folders(folders, destination_subfolder):
    for folder in folders:
        src_folder_path = os.path.join(source_folder, folder)
        dest_folder_path = os.path.join(destination_folder, destination_subfolder, folder)
        
        # Copy the entire directory contents
        shutil.copytree(src_folder_path, dest_folder_path)
        
        # Remove the original folder after copying
        shutil.rmtree(src_folder_path)

# Move folders to their respective subfolders
move_folders(train_folders, 'train')
move_folders(test_folders, 'test')
move_folders(validate_folders, 'validate')

print("Folders split and moved successfully.")
