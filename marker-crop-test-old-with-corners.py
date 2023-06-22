import os
import cv2
from cv2 import aruco
import numpy as np
from PIL import Image, ImageDraw

from lib.lego_colors import lego_colors_by_id
from lib.aruco_utils import aruco_ids_to_color_id, draw_aruco_corners
from lib.image_utils import image_to_data_url, correct_image_orientation, compute_image_hash
from lib.bounding_box import BoundingBox


class ArucoMarker:
    def __init__(self, id, bounding_box):
        self.id = id
        self.bounding_box = bounding_box

    @classmethod
    def from_aruco_detection(cls, id, points):
        return cls(
            id=id[0],
            bounding_box=BoundingBox.from_aruco(points[0])
        )

    def __repr__(self):
        return f"ArucoMarker({self.id}, {self.bounding_box})"

class ArucoMarkerSet:
    def __init__(self, markers):
        self.markers = markers

    @classmethod
    def from_aruco_detection(cls, ids, aruco_points):
        markers = [ArucoMarker.from_aruco_detection(id, point) for id, point in zip(ids, aruco_points)]
        return cls(markers)

    # Combine any where the centers are within threshold pixels of each other
    def combine_nearby_corner_markers(self, threshold):
        combined_markers = []
        for marker in self.markers:
            if marker.is_corner:
                # find index of nearby corner marker
                index = next((i for i, combined_marker in enumerate(combined_markers) if marker.is_nearby(combined_marker, threshold) and marker.is_corner), None)
                if index is None:
                    combined_markers.append(marker)
                else:
                    combined_markers[index] = combined_markers[index].combine(marker)
            else:
                combined_markers.append(marker)
        return ArucoMarkerSet(combined_markers)

    @property
    def corner_markers(self):
        return [marker for marker in self.markers if marker.is_corner]

    @property
    def color_markers(self):
        return [marker for marker in self.markers if marker.is_color]

    def valid(self):
        found_corners = len(self.corner_markers) == 4 or len(self.corner_markers) == 0
        found_colors = len(self.color_markers) == 2
        return found_corners and found_colors

    def cropping_box(self):
        # find the global center of the 4 corner markers

        center = (self.image_width/2, self.image_height/2)
        points = [marker.bounding_box.corner_closest_to(center) for marker in self.corner_markers ]
        # find the bounding box that fits within the points

        return points

    def __repr__(self):
        return f"ArucoMarkerSet({self.markers.__repr__()})"

dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
aruco_detector = aruco.ArucoDetector(dict)


image = Image.open("tmp/marker-crop-test.jpeg")
format = image.format.lower()

image = correct_image_orientation(image)
opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
aruco_points, ids, rejectedImgPoints = aruco_detector.detectMarkers(opencv_image)

# zip aruco corners and ids together
# markers = [ArucoMarker.from_aruco_detection(id, point) for id, point in zip(ids, aruco_points)]
# marker_set = ArucoMarkerSet(markers)
marker_set = ArucoMarkerSet.from_aruco_detection(ids, aruco_points)
combined_marker_set = marker_set.combine_nearby_corner_markers(threshold=0.1*image.width)

print(marker_set)
print("")
print(combined_marker_set)
print("")
print(combined_marker_set.cropping_box())



exit()

# Find the corner symbols. These are the symbols with value 99.
# Since we are scanning 3d boxes with the symbol on all sides, we combine
# any symbols that are close together.
corner_boxes = []
for index, id in enumerate(ids):
    if id == 99:
        corner_boxes.append(BoundingBox.from_aruco(aruco_corners[index][0]))
corner_boxes = BoundingBox.combine_nearby(corner_boxes, threshold=0.1*image.width)
found_corners_or_no_corners = len(corner_boxes) == 4 or len(corner_boxes) == 0

# Look for color symbols
color_ids = [id for id in ids if id <= 90]
found_colors = len(color_ids) == 2

# Draw our findings on a copy
image_copy = image.copy()
draw = ImageDraw.Draw(image_copy)
draw_aruco_corners(draw, aruco_corners, color='red', width=5)
for box in corner_boxes:
    box.grow(10).draw(draw, color='yellow', width=5)
image_copy.convert("RGB").save(f'tmp/last-capture-detect.{format}')

# if not found_corners_or_no_corners :
#     return jsonify({'success': False, 'message': f"Found {len(corner_boxes)} corner markers"})

# if not found_colors :
#     return jsonify({'success': False, 'message': f"Found {len(color_ids)} color markers"})

# Crop out any Aruco markers (corners and color markers)
all_boxes = [BoundingBox.from_aruco(aruco_corner[0]) for aruco_corner in aruco_corners]
image_center = (image.width / 2, image.height / 2)
x1 = max([box.x for box in all_boxes if box.center[0] < image_center[0]], default=0)
x2 = min([box.x for box in all_boxes if box.center[0] > image_center[0]], default=image.width)
y1 = max([box.y for box in all_boxes if box.center[1] < image_center[1]], default=0)
y2 = min([box.y for box in all_boxes if box.center[1] > image_center[1]], default=image.height)
cropping_box = BoundingBox(x1, y1, x2, y2)
cropped_image = cropping_box.crop(image)

color_id = aruco_ids_to_color_id(color_ids)
color = lego_colors_by_id[color_id]
print(f"Found color id {color_id} from {color_ids}")
