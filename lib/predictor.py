import torch

from lib.lego_colors import lego_colors_by_id
from lib.db import Db
from lib.compensate import canonical_part_id

class Predictor:
    def __init__(self, classification_model, color_model, db):
        self.classification_model = classification_model
        self.color_model = color_model
        self.db = db

    def predict_parts_and_colors(self, image):
        results = self.classification_model.predict(source=image)

        result = results[0].cpu()
        topk_values, topk_indices = torch.topk(result.probs, k=3)
        topk_classes = [result.names[i.item()] for i in topk_indices]

        color_results = self.color_model.predict(source=image)
        color_result = color_results[0].cpu()
        color_topk_values, topk_indices = torch.topk(color_result.probs, k=1)
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

    def part_name_or_blank(self, num):
        part = self.db.get_part_by_num(num)
        return part.name if part else '??? mismatched ids'
