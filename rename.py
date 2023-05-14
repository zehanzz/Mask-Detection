import os

with_masks_folder = 'dataset/train/with_mask'
without_masks_folder = 'dataset/train/without_mask'

with_masks_files = os.listdir(with_masks_folder)
for filename in with_masks_files:
    if filename.endswith(".jpg"):
        if filename.startswith("augmented"):
            new_filename = "_".join(filename.split("_")[:3])
            os.rename(filename, new_filename)
            old_file_path = os.path.join(with_masks_folder, filename)
            new_file_path = os.path.join(with_masks_folder, new_filename)
            os.rename(old_file_path, new_file_path)
        else:
            new_filename = filename.split("_")[0]
            old_file_path = os.path.join(with_masks_folder, filename)
            new_file_path = os.path.join(with_masks_folder, new_filename)
            os.rename(old_file_path, new_file_path)

# Preprocess images without masks
without_masks_files = os.listdir(without_masks_folder)
for filename in without_masks_files:
    if filename.endswith(".jpg"):
        if filename.startswith("augmented"):
            new_filename = "_".join(filename.split("_")[:3])
            os.rename(filename, new_filename)
            old_file_path = os.path.join(without_masks_folder, filename)
            new_file_path = os.path.join(without_masks_folder, new_filename)
            os.rename(old_file_path, new_file_path)
        else:
            new_filename = filename.split("_")[0]
            old_file_path = os.path.join(without_masks_folder, filename)
            new_file_path = os.path.join(without_masks_folder, new_filename)
            os.rename(old_file_path, new_file_path)