import os
import shutil
import random

# Define paths
source_folders = {
    'at_camera': 'Duncan/at_camera',
    'away': 'Duncan/away'
}
destination_folders = {
    'at_camera': 'Duncan/at_camera_split',
    'away': 'Duncan/away_split'
}
subfolders = ['train', 'test', 'validate']

# Create destination subfolders if they do not exist
for key in destination_folders:
    for subfolder in subfolders:
        os.makedirs(os.path.join(destination_folders[key], subfolder), exist_ok=True)

# List all subfolders in the source folders
image_folders = os.listdir(source_folders['at_camera'])
image_folders = [f for f in image_folders if os.path.isdir(os.path.join(source_folders['at_camera'], f))]

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
def move_folders(folders, destination_subfolder, source_key):
    for folder in folders:
        src_folder_path = os.path.join(source_folders[source_key], folder)
        dest_folder_path = os.path.join(destination_folders[source_key], destination_subfolder, folder)
        
        # Copy the entire directory contents
        shutil.copytree(src_folder_path, dest_folder_path)
        
        # Remove the original folder after copying
        shutil.rmtree(src_folder_path)

# Move folders to their respective subfolders for both at_camera and away
for split, folders_split in zip(subfolders, [train_folders, test_folders, validate_folders]):
    move_folders(folders_split, split, 'at_camera')
    move_folders(folders_split, split, 'away')

# Replace source folders with split folders
for key in source_folders:
    shutil.rmtree(source_folders[key])
    shutil.move(destination_folders[key], source_folders[key])

print("Folders split and moved successfully.")
