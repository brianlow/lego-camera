import os
from pathlib import Path
from lib.lego_colors import lego_colors_by_id

# Dangerous!
###
### def rename_and_move_files(path):
###     for file_path in Path(path).rglob('*'):
###         if file_path.is_file():
###             filename = file_path.name
###             ext = file_path.suffix
###             if ext.lower() in [".jpg", ".png", ".jpeg"]:
###                 id = filename.split('.')[1]
###                 color = lego_colors_by_id[int(id)]
###                 if not color:
###                     raise Exception(f"Color not found for id {id} for file {file_path}")
###                 new_filename = f"{color.name.replace(' ', '')}.{id}{ext}"
###                 new_dir = Path(path) / id
###                 new_dir.mkdir(exist_ok=True)
###                 new_file_path = new_dir / new_filename
###                 if file_path == new_file_path:
###                     print(f"Skipping {file_path}")
###                     continue
###                 if new_file_path.exists():
###                     raise Exception(f"File already exists: {new_file_path}")
###                 print(f"Moving {file_path} to {new_file_path}")
###                 file_path.rename(new_file_path)
###
# Running the rename function on your directory.
rename_and_move_files("tmp/colors")
