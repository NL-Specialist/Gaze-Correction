import os
import shutil
import random
import tempfile

async def split_folders(input_path):
    try:
        # Define paths
        source_folders = {
            'at_camera': os.path.join(input_path, 'at_camera'),
            'away': os.path.join(input_path, 'away')
        }
        destination_folders = {
            'at_camera': tempfile.mkdtemp(),
            'away': tempfile.mkdtemp()
        }
        subfolders = ['train', 'test', 'validate']

        # Create destination subfolders if they do not exist
        for key in destination_folders:
            for subfolder in subfolders:
                os.makedirs(os.path.join(destination_folders[key], subfolder), exist_ok=True)

        # Function to move folders
        def move_folders(folders, destination_subfolder, source_key):
            try:
                for folder in folders:
                    src_folder_path = os.path.join(source_folders[source_key], folder)
                    dest_folder_path = os.path.join(destination_folders[source_key], destination_subfolder, folder)
                    
                    # Copy the entire directory contents
                    shutil.copytree(src_folder_path, dest_folder_path)
                    
                    # Remove the original folder after copying
                    shutil.rmtree(src_folder_path)
            except Exception as e:
                print(f"Error while moving folders for {source_key} to {destination_subfolder}: {e}")
                raise

        total_image_folders = 0
        
        # Ensure the number of folders in at_camera and away match
        at_camera_folders = os.listdir(source_folders['at_camera'])
        at_camera_folders = [f for f in at_camera_folders if os.path.isdir(os.path.join(source_folders['at_camera'], f))]

        away_folders = os.listdir(source_folders['away'])
        away_folders = [f for f in away_folders if os.path.isdir(os.path.join(source_folders['away'], f))]

        if len(at_camera_folders) != len(away_folders):
            print(f"Mismatch in folder counts: {len(at_camera_folders)} in 'at_camera' vs {len(away_folders)} in 'away'. Exiting.")
            return

        # Shuffle folder names to randomize split, but do it once for both
        combined_folders = list(zip(at_camera_folders, away_folders))
        random.shuffle(combined_folders)

        total_folders = len(combined_folders)
        train_split = int(0.7 * total_folders)
        test_split = int(0.2 * total_folders)
        validate_split = total_folders - train_split - test_split

        # Split the folders into train, test, and validate
        train_folders = combined_folders[:train_split]
        test_folders = combined_folders[train_split:train_split + test_split]
        validate_folders = combined_folders[train_split + test_split:]

        # Move folders to respective subfolders for both at_camera and away
        for (at_folder, away_folder) in train_folders:
            move_folders([at_folder], 'train', 'at_camera')
            move_folders([away_folder], 'train', 'away')

        for (at_folder, away_folder) in test_folders:
            move_folders([at_folder], 'test', 'at_camera')
            move_folders([away_folder], 'test', 'away')

        for (at_folder, away_folder) in validate_folders:
            move_folders([at_folder], 'validate', 'at_camera')
            move_folders([away_folder], 'validate', 'away')

        # Replace source folders with the split folders
        for key in source_folders:
            shutil.rmtree(source_folders[key])
            shutil.move(destination_folders[key], source_folders[key])

        print("Folders split and moved successfully.")
    
    except Exception as e:
        print(f"An error occurred in the split_folders function: {e}")
        raise

# Usage
# await split_folders('datasets/Duncan')
