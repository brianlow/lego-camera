import os
import uuid
from ultralytics import YOLO
from PIL import Image
from flask import Flask, request, jsonify, g
from flask_cors import CORS
import base64
import torch
from io import BytesIO

from PIL import Image
from lib.lego_colors import lego_colors_by_id
from lib.bounding_box import BoundingBox
from lib.json_utils import decimal_default
from lib.image_utils import image_to_data_url, correct_image_orientation
from lib.db import Db
from lib.compensate import canonical_part_id
from lib.predictor import Predictor

detection_model = YOLO(
    "detect-10-4k-real-and-renders-nano-1024-image-size2.pt")
classification_model = YOLO("03-447x.pt")
color_model = YOLO("color-03-common-5k-3-camera-blues.pt")


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
    try:
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
        # print("--- response:")
        # r = response.copy()
        # r['source_url'] = r['source_url'][:15] + "..."
        # print(r)

        return jsonify(response)

    except Exception as e:
        # If there was an error processing the image, return an error message
        response = {'success': False,
                    'message': 'Error processing image: {}'.format(str(e))}
        return jsonify(response, default=decimal_default)

@app.teardown_appcontext
def close_connection(exception):
    db.close()

if __name__ == '__main__':
    print("-----------")
    for rule in app.url_map.iter_rules():
        methods = ','.join(sorted(rule.methods))
        print(f"{rule} ({methods})")
    app.run(port=8000, debug=True)
