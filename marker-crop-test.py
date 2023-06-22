import os
from PIL import Image, ImageDraw
from ultralytics import YOLO

from lib.lego_colors import lego_colors_by_id
from lib.image_utils import correct_image_orientation
from lib.bounding_box import BoundingBox
from lib.predictor import Predictor
from lib.aruco_marker_set import ArucoMarkerSet


detection_model = YOLO("lego-detect-13-7k-more-negatives3.pt")
color_model = YOLO("lego-color-10-more-photos-nano.pt")
predictor = Predictor(detection_model, None, color_model, None)


print("Opening image...")
image = Image.open("tmp/IMG_7936.jpg")
# image = Image.open("tmp/marker-crop-test-3.jpeg")
format = image.format.lower()
image = correct_image_orientation(image)

print("Looking for Aruco markers...")
marker_set = ArucoMarkerSet.detect_from_image(image)

print("Looking for Lego pieces...")
pieces = predictor.detect_objects(image)

print("Calculating cropping box...")
min_x = min([piece.x1 for piece in pieces])
max_x = max([piece.x2 for piece in pieces])
min_y = min([piece.y1 for piece in pieces])
max_y = max([piece.y2 for piece in pieces])
cropping_box = BoundingBox(min_x, min_y, max_x, max_y)
cropping_box = cropping_box.grow(max(image.width, image.height)*0.025)
for marker in marker_set.markers:
    cropping_box = cropping_box.shrink_from(marker.bounding_box.grow(marker.bounding_box.width*0.1))

print("Detecting colors...")
actual_color = None
piece_colors = []
if marker_set.valid:
    actual_color = lego_colors_by_id[marker_set.color_id]

    for piece in pieces:
        predicted_color, confidence = predictor.predict_color(piece.crop(image))
        piece_colors.append((piece, predicted_color, confidence))


print("Drawing results...")
image_copy = image.copy()
draw = ImageDraw.Draw(image_copy)
marker_set.draw(draw, color='red', width=5)
cropping_box.draw(draw, color='green', width=5)
if not actual_color is None:
    for piece, predicted_color, confidence in piece_colors:
        if not piece.is_inside(cropping_box):
            continue

        correct = predicted_color == actual_color
        piece.draw(draw, color='white', width=10)
        piece.draw_label(draw, f"{confidence * 100:.0f}%: {predicted_color.name} ({predicted_color.id})",
                        text_color = 'black' if correct else 'red',
                        swatch_color=predicted_color.hex())


new_filename = f"tmp/marker-crop-test-3-out.{format}"
image_copy.save(new_filename)

print(f"Saved to {new_filename}")
