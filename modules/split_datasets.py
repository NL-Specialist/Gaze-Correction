import os
import shutil
import random

def split_folders(input_path):
    # Define paths
    source_folders = {
        'at_camera': os.path.join(input_path, 'at_camera'),
        'away': os.path.join(input_path, 'away')
    }
    destination_folders = {
        'at_camera': os.path.join(input_path, 'at_camera_split'),
        'away': os.path.join(input_path, 'away_split')
    }
    subfolders = ['train', 'test', 'validate']

    # Create destination subfolders if they do not exist
    for key in destination_folders:
        for subfolder in subfolders:
            os.makedirs(os.path.join(destination_folders[key], subfolder), exist_ok=True)

    # List all subfolders in the source folders
    image_folders = os.listdir(source_folders['at_camera'])
    image_folders = [f for f in image_folders if os.path.isdir(os.path.join(source_folders['at_camera'], f))]

    # Print the initial count of image folders
    print(f"Total image folders found: {len(image_folders)}")
    
    # Shuffle image folders to randomize the split
    random.shuffle(image_folders)

    # Calculate split indices
    total_folders = len(image_folders)
    train_split = int(0.7 * total_folders)
    test_split = int(0.2 * total_folders)
    validate_split = total_folders - train_split - test_split

    # Print the split indices
    print(f"Total folders: {total_folders}")
    print(f"Train split: {train_split}")
    print(f"Test split: {test_split}")
    print(f"Validate split: {validate_split}")

    # Split folders
    train_folders = image_folders[:train_split]
    test_folders = image_folders[train_split:train_split + test_split]
    validate_folders = image_folders[train_split + test_split:]

    # Print lengths of each split
    print(f"Train folders count: {len(train_folders)}")
    print(f"Test folders count: {len(test_folders)}")
    print(f"Validate folders count: {len(validate_folders)}")

    # Function to move folders
    def move_folders(folders, destination_subfolder, source_key):
        print(f"Moving {len(folders)} folders to {destination_subfolder}/{source_key}")
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

# Usage
# split_folders('datasets/Duncan')
