import torch
from PIL import Image

from lib.bounding_box import BoundingBox
from lib.lego_colors import lego_colors_by_id
from lib.db import Db
from lib.compensate import canonical_part_id

class Predictor:
    def __init__(self, detection_model, classification_model, color_model, db):
        self.detection_model = detection_model
        self.classification_model = classification_model
        self.color_model = color_model
        self.db = db

    def detect_objects(self, image):
        results = self.detection_model(image.convert("RGB"))

        if len(results) == 0:
            return []

        Image.fromarray(
            results[0].cpu().plot()[..., ::-1]
        ).save('tmp/last-classify-detection.jpeg')
        boxes = [BoundingBox.from_yolo(yolo_box)
                    for yolo_box in results[0].cpu().boxes]
        boxes = list(filter(lambda box: not box.is_touching_frame(
            image.width, image.height), boxes))
        boxes.sort(key=lambda box: box.area, reverse=True)

        return boxes


    def predict_parts_and_colors(self, image):
        image.convert("RGB").save('tmp/last-classify-transform.jpg')

        results = self.classification_model.predict(source=image)

        result = results[0].cpu()
        print("---- probs")
        print(result.probs)
        print("---- result")
        print(result)
        topk_values, topk_indices = torch.topk(result.probs.data, k=3)
        topk_classes = [result.names[i.item()] for i in topk_indices]

        color_results = self.color_model.predict(source=image.convert("RGB"))
        color_result = color_results[0].cpu()
        color_topk_values, topk_indices = torch.topk(color_result.probs.data, k=1)
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
            ldraw_id = self.db.get_ldraw_id_for_part_num(part_num)

            parts.append({
                'id': part_num,
                'name': self.part_name_or_blank(part_num),
                'url': f"/images/{ldraw_id}.png",
                'confidence': confidence,
                'color': {
                    'id': predicted_color.id,
                    'name': predicted_color.name,
                    'hex': f"#{predicted_color.hex()}",
                    'confidence': predicted_color_confidence,
                }

            })

        return parts

    def predict_color(self, image):
        results = self.color_model.predict(source=image.convert("RGB"))
        result = results[0].cpu()
        topk_values, topk_indices = torch.topk(result.probs.data, k=1)
        topk_classes = [result.names[i.item()] for i in topk_indices]
        predicted_color = lego_colors_by_id[int(topk_classes[0])]
        predicted_confidence = float(topk_values[0])
        return predicted_color, predicted_confidence

    def part_name_or_blank(self, num):
        part = self.db.get_part_by_num(num)
        return part.name if part else '??? mismatched ids'
