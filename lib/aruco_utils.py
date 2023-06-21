
def aruco_ids_to_color_id(ids):
    color_id = aruco_id_to_color_id(ids[0])
    color_id += aruco_id_to_color_id(ids[1])
    return color_id

def aruco_id_to_color_id(id):
    if id < 50:
        return id
    else:
        return (id-50) * 50

def draw_aruco_corners(draw, corners, color, width):
    for corner in corners:
        points = [tuple(point) for point in corner[0].astype(int)]  # convert corners to integer
        draw.line((*points[0], *points[1], *points[2], *points[3], *points[0]), fill=color, width=width)
