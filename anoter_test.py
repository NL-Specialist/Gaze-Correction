import os
import re

# Define the folder path
folder_path = r"D:\OneDrive - North-West University\Outer Aspect\Mr. Temple\LAW\NZ Arms Act  & 3egs for AI\Arms Act LAm Expert Training\Arms Act LAm Expert Training"

# General regular expression to match the filename format
file_pattern = re.compile(r"-v-([^-]+)-(.*)--(\d{4})-NZDC-(\d+)\.pdf")

# Iterate over all files in the directory
for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        match = file_pattern.search(filename)
        if match:
            opponent = match.group(1).strip()
            person_name = match.group(2).strip()
            year = match.group(3)
            case_id = match.group(4)
            
            # Construct the new filename
            new_filename = f"{person_name}-v-{opponent}-{year}-NZDC-{case_id}.pdf"
            
            # Get full file paths
            old_file = os.path.join(folder_path, filename)
            new_file = os.path.join(folder_path, new_filename)
            
            # Rename the file
            os.rename(old_file, new_file)
            print(f"Renamed: {old_file} -> {new_file}")
        else:
            print(f"Skipping: {filename} (No match found)")

