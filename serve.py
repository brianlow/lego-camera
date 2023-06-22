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
from lib.aruco_marker_set import ArucoMarkerSet

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

        image_copy.convert("RGB").save(f'tmp/last-capture-detect.{format}')

        print("Saving processed file...")
        cropped_image = cropping_box.crop(image)
        color_name = re.sub(r'\W+', '', actual_color.name)
        hash = compute_image_hash(cropped_image)
        new_filename = f"tmp/colors/{color_name}-{hash[:6]}.{actual_color.id}.{format}"
        cropped_image.save(new_filename)

        response = {
            'success': actual_color is not None,
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
