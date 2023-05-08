from ultralytics import YOLO
from PIL import Image
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import torch
from io import BytesIO
from lib.lego_colors import lego_colors_by_id

from typing import Callable, Tuple, Union
from dataclasses import dataclass
from PIL import Image
from lib.bounding_box_funcs import make_bbox_square, convert_xywh_to_upper_left_xy

# model = YOLO("10-10x-square.pt")  # 10 classes
# detection_model = YOLO("detect-05-sample-real3.pt")
# detection_model = YOLO("detect-07-4k-real-and-renders.pt")
detection_model = YOLO("detect-10-4k-real-and-renders-nano-1024-image-size2.pt")
classification_model = YOLO("03-447x.pt")
color_model = YOLO("color-03-common-nano.pt")


# Images are already closely cropped and I think there is valuable
# information at the edges. So make images square so we can just
# resize rather than resize + crop.
class SquareImageTransform:
    def __init__(self, fill_color=(255, 255, 255)):
        self.fill_color = fill_color

    def __call__(self, image):
        width, height = image.size
        new_size = max(width, height)
        new_image = Image.new('RGB', (new_size, new_size), self.fill_color)
        new_image.paste(image, ((new_size - width) // 2, (new_size - height) // 2))
        return new_image

class ResizeTransform:
    def __init__(self, length=224):
      self.length = length

    def __call__(self, image):
      image = image.resize((self.length, self.length))
      return image



square = SquareImageTransform()
resize = ResizeTransform()

def decimal_default(obj):
    if isinstance(obj, (float, complex)):
        return format(obj, '.3f')
    return str(obj)


app = Flask(__name__, static_url_path='/')

app.static_folder = 'static'

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
            boxes = list(map(convert_xywh_to_upper_left_xy, results[0].cpu().boxes))
            Image.fromarray(results[0].cpu().plot()).save('tmp/last-detect-detection.png')

        response = {
            'boxes': boxes
        }
        print("---")
        print(response)

        return jsonify(response)

    except Exception as e:
        # If there was an error processing the image, return an error message
        response = {'success': False, 'message': 'Error processing image: {}'.format(str(e))}
        return jsonify(response, default=decimal_default)

@app.route('/classify', methods=['POST'])
def classify():
    try:
        encoded_image = request.json['image']
        image_bytes = base64.b64decode(encoded_image)
        image = Image.open(BytesIO(image_bytes))

        image.convert("RGB").save('tmp/last-classify-original.png')

        results = detection_model(image.convert("RGB"))
        boxes = []
        boxes_json = []
        if len(results) > 0:
            boxes = results[0].cpu().boxes
            boxes_json = list(map(lambda box: { "x": box.xyxy[0][0].item(), "y": box.xyxy[0][1].item(), "x2": box.xyxy[0][2].item(), "y2": box.xyxy[0][3].item()}, boxes))
            Image.fromarray(results[0].cpu().plot()).save('tmp/last-classify-detection.png')

        if len(boxes) > 0:
            sorted_boxes = sorted(boxes, key=lambda box: box.xywh[0][2].item() * box.xywh[0][3].item(), reverse=True)
            box = sorted_boxes[0]
            x1, y1, x2, y2 = make_bbox_square(box.xyxy[0][0].item(), box.xyxy[0][1].item(), box.xyxy[0][2].item(), box.xyxy[0][3].item())
            image = image.crop((x1, y1, x2, y2))

        image.convert("RGB").save('tmp/last-classify-transform.png')

        results = classification_model.predict(source=image)
        result = results[0].cpu()
        topk_values, topk_indices = torch.topk(result.probs, k=3)
        topk_classes = [result.names[i.item()] for i in topk_indices]

        color_results = color_model.predict(source=image)
        color_result = color_results[0].cpu()
        color_topk_values, topk_indices = torch.topk(color_result.probs, k=3)
        color_topk_classes = [color_result.names[i.item()] for i in topk_indices]
        predicted_color = lego_colors_by_id[int(color_topk_classes[0])]
        predicted_color_confidence = float(color_topk_values[0])

        response = {
            'classes': [{ 'label': topk_classes[i], 'probability': round(topk_values[i].item() * 100, 0) } for i in range(len(topk_classes))],
            'boxes': boxes_json,
            'color': {
                'id': predicted_color.id,
                'name': predicted_color.name,
                'hex': f"#{predicted_color.hex()}",
                'confidence': predicted_color_confidence,
            }
        }
        print("---")
        print(response)
        return jsonify(response)

    except Exception as e:
        # If there was an error processing the image, return an error message
        response = {'success': False, 'message': 'Error processing image: {}'.format(str(e))}
        return jsonify(response, default=decimal_default)

if __name__ == '__main__':
    print("-----------")
    for rule in app.url_map.iter_rules():
        methods = ','.join(sorted(rule.methods))
        print(f"{rule} ({methods})")
    app.run(port=8000, debug=True)
