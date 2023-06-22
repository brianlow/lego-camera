import os
from pathlib import Path
import cv2
from cv2 import aruco
import shutil
import numpy as np
from PIL import Image
from lib.lego_colors import lego_colors_by_id
from lib.image_utils import compute_image_hash
from lib.aruco_utils import aruco_ids_to_color_id, draw_aruco_corners
from lib.bounding_box import BoundingBox

aruco_detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_6X6_100))


def add_hash_to_filenames(source, destination):
    for root, dirs, files in os.walk(source):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                filename = os.path.join(root, file)
                print(f"Processing {filename}")

                (prefix, id, ext) = os.path.basename(filename).split(".")

                image = Image.open(filename)
                opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                aruco_corners, ids, rejectedImgPoints = aruco_detector.detectMarkers(opencv_image)
                boxes = [BoundingBox.from_aruco(aruco_corner[0]) for aruco_corner in aruco_corners]

                image_center = (image.width / 2, image.height / 2)
                x1 = max([box.x for box in boxes if box.center[0] < image_center[0]], default=0)
                x2 = min([box.x for box in boxes if box.center[0] > image_center[0]], default=image.width)
                y1 = max([box.y for box in boxes if box.center[1] < image_center[1]], default=0)
                y2 = min([box.y for box in boxes if box.center[1] > image_center[1]], default=image.height)

                print(f"cropping to x: {x1}, {x2}, y: {y1}, {y2}")
                crop = BoundingBox(x1, y1, x2, y2)
                cropped_image = crop.crop(image)

                new_filename = f"{prefix}.{id}.{ext}"
                new_filename = os.path.join(destination, id, new_filename)
                print(f"Copying {filename}   ->    {new_filename}")

                os.makedirs(os.path.dirname(new_filename), exist_ok=True)

                if os.path.exists(new_filename):
                    raise Exception(f"destination file already exists {new_filename}")

                cropped_image.save(new_filename)


add_hash_to_filenames("tmp/colors", "tmp/colors-cropped")
