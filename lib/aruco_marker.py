from lib.bounding_box import BoundingBox

class ArucoMarker:
    def __init__(self, id, bounding_box):
        self.id = id
        self.bounding_box = bounding_box

    @classmethod
    def from_aruco_detection(cls, id, points):
        return cls(
            id=id[0],
            bounding_box=BoundingBox.from_aruco(points[0])
        )

    def __repr__(self):
        return f"ArucoMarker({self.id}, {self.bounding_box})"
