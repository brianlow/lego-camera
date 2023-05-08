# Input is xywh format, output is xyxy
def make_bbox_square(x1, y1, x2, y2):
    width = x2 - x1
    height = y2 - y1

    max_dim = max(width, height)
    width_diff = max_dim - width
    height_diff = max_dim - height

    x1_new = x1 - width_diff / 2
    x2_new = x1_new + max_dim
    y1_new = y1 - height_diff / 2
    y2_new = y1_new + max_dim

    return x1_new, y1_new, x2_new, y2_new

def convert_xywh_to_upper_left_xy(box):
    x_center = box.xywh[0][0].item()
    y_center = box.xywh[0][1].item()
    w = box.xywh[0][2].item()
    h = box.xywh[0][3].item()

    x_upper_left = x_center - w / 2
    y_upper_left = y_center - h / 2

    return {"x": x_upper_left, "y": y_upper_left, "w": w, "h": h}
