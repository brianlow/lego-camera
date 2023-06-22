import cv2
import numpy as np
from cv2 import aruco

from lib.aruco_utils import aruco_ids_to_color_id
from lib.aruco_marker import ArucoMarker

class ArucoMarkerSet:
    def __init__(self, markers):
        self.markers = markers

    @classmethod
    def from_aruco_detection(cls, ids, aruco_points):
        if ids is None:
            return cls([])

        markers = [ArucoMarker.from_aruco_detection(id, point) for id, point in zip(ids, aruco_points)]
        return cls(markers)

    @classmethod
    def detect_from_image(cls, image):
        dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
        aruco_detector = aruco.ArucoDetector(dict)
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        aruco_points, ids, _ = aruco_detector.detectMarkers(opencv_image)
        return ArucoMarkerSet.from_aruco_detection(ids, aruco_points)

    @property
    def valid(self):
        return len(self.markers) == 2

    @property
    def color_id(self):
        if self.valid:
            return aruco_ids_to_color_id([marker.id for marker in self.markers])
        else:
            return None

    def draw(self, draw, color='red', width=5):
        for marker in self.markers:
            marker.bounding_box.draw(draw, color=color, width=width)

    def __repr__(self):
        return f"ArucoMarkerSet({self.markers.__repr__()})"
