import os
from pathlib import Path
import shutil
from PIL import Image
from lib.lego_colors import lego_colors_by_id
from lib.image_utils import compute_image_hash

def add_hash_to_filenames(source, destination):
    for root, dirs, files in os.walk(source):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                filename = os.path.join(root, file)

                (prefix, id, ext) = os.path.basename(filename).split(".")

                image = Image.open(filename)
                hash = compute_image_hash(image)

                new_filename = f"{prefix}-{hash[:6]}.{id}.{ext}"
                new_filename = os.path.join(destination, id, new_filename)
                print(f"Copying {filename}   ->    {new_filename}")

                os.makedirs(os.path.dirname(new_filename), exist_ok=True)

                # copy filename to new_filename
                shutil.copyfile(filename, new_filename)


add_hash_to_filenames("tmp/colors-old", "tmp/colors-old-renamed")
