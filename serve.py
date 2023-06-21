import os
import uuid
from ultralytics import YOLO
from PIL import Image, ImageDraw
from flask import Flask, request, jsonify, g
from flask_cors import CORS
import base64
import torch
from io import BytesIO
import cv2
from cv2 import aruco
import numpy as np
import traceback
import re
import hashlib

from PIL import Image
from lib.lego_colors import lego_colors_by_id
from lib.bounding_box import BoundingBox
from lib.json_utils import decimal_default
from lib.image_utils import image_to_data_url, correct_image_orientation, compute_image_hash
from lib.db import Db
from lib.compensate import canonical_part_id
from lib.predictor import Predictor
from lib.aruco_utils import aruco_ids_to_color_id, draw_aruco_corners

detection_model = YOLO("lego-detect-13-7k-more-negatives3.pt")
classification_model = YOLO("03-447x.pt")
color_model = YOLO("lego-color-10-more-photos-nano.pt")

aruco_detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_6X6_100))


app = Flask(__name__, static_url_path='/')

app.static_folder = 'static'

db = Db(g)

predictor = Predictor(detection_model, classification_model, color_model, db)

@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/classes', methods=['GET'])
def get_classes():
    print(list(classification_model.names.values()))
    return jsonify({'classes': list(classification_model.names.values())})

# For data capture
@app.route('/capture', methods=['POST'])
def capture():
    try:
        encoded_image = request.json['image']
        image_bytes = base64.b64decode(encoded_image)
        image = Image.open(BytesIO(image_bytes))
        format = image.format.lower()

        image.convert("RGB").save(f'tmp/last-capture-original.{format}')

        image = correct_image_orientation(image)
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        aruco_corners, ids, rejectedImgPoints = aruco_detector.detectMarkers(opencv_image)

        ids = [list[0] for list in ids]
        print(f"Found ids: {ids}")

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

        if not found_corners_or_no_corners :
            return jsonify({'success': False, 'message': f"Found {len(corner_boxes)} corner markers"})

        if not found_colors :
            return jsonify({'success': False, 'message': f"Found {len(color_ids)} color markers"})

        # Crop out any Aruco markers (corners and color markers)
        all_boxes = [BoundingBox.from_aruco(aruco_corner[0]) for aruco_corner in aruco_corners]
        image_center = (image.width / 2, image.height / 2)
        x1 = max([box.x for box in all_boxes if box.center[0] < image_center[0]], default=0)
        x2 = min([box.x for box in all_boxes if box.center[0] > image_center[0]], default=image.width)
        y1 = max([box.y for box in all_boxes if box.center[1] < image_center[1]], default=0)
        y2 = min([box.y for box in all_boxes if box.center[1] > image_center[1]], default=image.height)
        cropping_box = BoundingBox(x1, y1, x2, y2)
        cropped_image = cropping_box.crop(image)
        pieces = predictor.detect_objects(cropped_image)
        print(f"Found {len(pieces)} lego pieces")

        color_id = aruco_ids_to_color_id(color_ids)
        color = lego_colors_by_id[color_id]
        print(f"Found color id {color_id} from {color_ids}")

        for piece in pieces:
            piece.move(cropping_box.x1, cropping_box.y1).draw(draw, color='purple', width=5)
        cropping_box.grow(20).draw(draw, color='white', width=5)
        cropping_box.grow(20).draw_label(draw, f"{color.id} - {color.name}", 'black', color.hex())
        image_copy.convert("RGB").save(f'tmp/last-capture-detect.{format}')

        color_name = re.sub(r'\W+', '', color.name)
        hash = compute_image_hash(image)
        new_filename = f"tmp/colors/{color_name}-{hash[:6]}.{color.id}.{format}"
        os.makedirs(os.path.dirname(new_filename), exist_ok=True)
        cropped_image.save(new_filename)

        response = {
            'color_id': color.id,
            'color_name': color.name,
            'color_hex': color.hex(),
            'corners': len(corner_boxes),
            'filename': new_filename,
        }

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        response = {'success': False, 'message': 'Error processing image: {}'.format(str(e))}
        return jsonify(response)


@app.route('/detect', methods=['POST'])
def detect():
    try:
        encoded_image = request.json['image']
        image_bytes = base64.b64decode(encoded_image)
        image = Image.open(BytesIO(image_bytes))

        image.convert("RGB").save('tmp/last-detect-original.png')

        results = detection_model(image.convert("RGB"))
        boxes = []
        if len(results) > 0:
            boxes = [BoundingBox.from_yolo(yolo_box)
                     for yolo_box in results[0].cpu().boxes]
            Image.fromarray(
                results[0].cpu().plot()[..., ::-1]
            ).save('tmp/last-detect-detection.png')

        response = {
            'boxes': [{"x": box.x, "y": box.y, "w": box.width, "h": box.height, "valid": not box.is_touching_frame(image.width, image.height)} for box in boxes]
        }
        print("---")
        print(response)

        return jsonify(response)

    except Exception as e:
        # If there was an error processing the image, return an error message
        response = {'success': False,
                    'message': 'Error processing image: {}'.format(str(e))}
        return jsonify(response, default=decimal_default)

@app.route('/classify', methods=['POST'])
def classify():
    # try:
    encoded_image = request.json['image']
    image_bytes = base64.b64decode(encoded_image)
    image = Image.open(BytesIO(image_bytes))
    print(f"----- format: {image.format}")
    image = correct_image_orientation(image)

    image.convert("RGB").save('tmp/last-classify-original.jpeg')

    boxes = predictor.detect_objects(image)
    boxes = boxes[:15] # sorted by size desc, limit predictions b/c slow

    objects = []
    for box in boxes:
        # TODO: square after cropping to avoid snagging other parts
        box_image = box.square().crop(image)
        parts = predictor.predict_parts_and_colors(box_image)
        objects.append({
        'source_url': image_to_data_url(box_image.convert("RGB")),
        'parts': parts,
    })

    color_id = request.args.get('color-id')
    print(f" ---  color_id: {color_id}")
    print(f" ---  color_id: {request.args}")
    print(f" ---  color_id: {request.args.keys()}")
    if not color_id is None:
        color = lego_colors_by_id[int(color_id)]
        os.makedirs(f'tmp/colors/{color.id}', exist_ok=True)
        image.convert("RGB").save(f'tmp/colors/{color.id}/{color.name.replace(" ", "")}-{str(uuid.uuid4())[:6]}.{color.id}.jpeg')

    response = {
        'objects': objects
    }
    print("--- response:")
    r = response.copy()
    # r['source_url'] = r['source_url'][:15] + "..."
    # print(r)

    return jsonify(response)

    # except Exception as e:
    #     # If there was an error processing the image, return an error message
    #     response = {'success': False,
    #                 'message': 'Error processing image: {}'.format(str(e))}
    #     return jsonify(response)

@app.teardown_appcontext
def close_connection(exception):
    db.close()

if __name__ == '__main__':
    print("-----------")
    for rule in app.url_map.iter_rules():
        methods = ','.join(sorted(rule.methods))
        print(f"{rule} ({methods})")
    app.run(port=8000, debug=True)
