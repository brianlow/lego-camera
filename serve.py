from ultralytics import YOLO
from PIL import Image
from flask import Flask, request, jsonify, g
from flask_cors import CORS
import base64
import torch
from io import BytesIO
from lib.lego_colors import lego_colors_by_id

from PIL import Image
from lib.bounding_box import BoundingBox
from lib.json_utils import decimal_default
from lib.image_utils import image_to_data_url, correct_image_orientation
from lib.db import Db
from lib.compensate import canonical_part_id

detection_model = YOLO(
    "detect-10-4k-real-and-renders-nano-1024-image-size2.pt")
classification_model = YOLO("03-447x.pt")
color_model = YOLO("color-03-common-nano.pt")


app = Flask(__name__, static_url_path='/')

app.static_folder = 'static'

db = Db(g)

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
        image = correct_image_orientation(image)

        image.convert("RGB").save('tmp/last-classify-original.jpeg')

        results = detection_model(image.convert("RGB"))

        boxes = []
        if len(results) > 0:
            Image.fromarray(
                results[0].cpu().plot()[..., ::-1]
            ).save('tmp/last-classify-detection.jpeg')
            boxes = [BoundingBox.from_yolo(yolo_box)
                     for yolo_box in results[0].cpu().boxes]
            boxes = list(filter(lambda box: not box.is_touching_frame(
                image.width, image.height), boxes))
            if len(boxes) > 0:
                largest_box = max(boxes, key=lambda box: box.area)
                image = largest_box.square().crop(image)

        image.convert("RGB").save('tmp/last-classify-transform.jpg')

        results = classification_model.predict(source=image)

        result = results[0].cpu()
        topk_values, topk_indices = torch.topk(result.probs, k=3)
        topk_classes = [result.names[i.item()] for i in topk_indices]

        color_results = color_model.predict(source=image)
        color_result = color_results[0].cpu()
        color_topk_values, topk_indices = torch.topk(color_result.probs, k=3)
        color_topk_classes = [color_result.names[i.item()]
                              for i in topk_indices]
        predicted_color = lego_colors_by_id[int(color_topk_classes[0])]
        predicted_color_confidence = float(color_topk_values[0])

        parts = []
        for i in range(len(topk_classes)):
            confidence = topk_values[i].item()
            if confidence < 0.10:
                continue
            part_num = canonical_part_id(topk_classes[i])
            ldraw_id = db.get_ldraw_id_for_part_num(part_num)

            parts.append({
                'id': part_num,
                'name': part_name_or_blank(part_num),
                'url': f"/images/{ldraw_id}.png",
                'confidence': confidence
            })

        response = {
            'source_url': image_to_data_url(image.convert("RGB")),
            'parts': parts,
            'color': {
                'id': predicted_color.id,
                'name': predicted_color.name,
                'hex': f"#{predicted_color.hex()}",
                'confidence': predicted_color_confidence,
            }
        }
        print("--- response:")
        r = response.copy()
        r['source_url'] = r['source_url'][:15] + "..."
        print(r)

        return jsonify(response)

    except Exception as e:
        # If there was an error processing the image, return an error message
        response = {'success': False,
                    'message': 'Error processing image: {}'.format(str(e))}
        return jsonify(response, default=decimal_default)

def part_name_or_blank(num):
    part = db.get_part_by_num(num)
    return part.name if part else '??? mismatched ids'

@app.teardown_appcontext
def close_connection(exception):
    db.close()

if __name__ == '__main__':
    print("-----------")
    for rule in app.url_map.iter_rules():
        methods = ','.join(sorted(rule.methods))
        print(f"{rule} ({methods})")
    app.run(port=8000, debug=True)
