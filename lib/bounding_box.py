from PIL import ImageFont
import os
import hashlib

class BoundingBox:
    _font = None

    def __init__(self,
    x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @classmethod
    def from_yolo(cls, yolo_box):
        return cls(
          x1=int(yolo_box.xyxy[0][0].int()),
          y1=int(yolo_box.xyxy[0][1].int()),
          x2=int(yolo_box.xyxy[0][2].int()),
          y2=int(yolo_box.xyxy[0][3].int())
        )

    @classmethod
    def from_xywh(cls, x, y, w, h):
        return cls(
          x1=x,
          y1=y,
          x2=x+w,
          y2=y+h
        )

    @classmethod
    def font(self):
        if self._font is None:
            font_path = os.path.expanduser('~/Library/Fonts/Arial.ttf')
            self._font = ImageFont.truetype(font_path, size=24)
        return self._font

    @property
    def x(self):
        return self.x1

    @property
    def y(self):
        return self.y1

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return self.width * self.height

    # Extracts a portion of an image
    def crop(self, image):
        return image.crop((int(self.x1), int(self.y1), int(self.x2), int(self.y2)))

    def draw(self, draw):
        coords = ((self.x1, self.y1), (self.x2, self.y2))
        draw.rectangle(coords, outline='white', width=2)

    # Draw a label below the box, colors are a hex string
    def draw_label(self, draw, text, text_color, swatch_color):
        x = self.x1 # bottom left corner of box, top left of label
        y = self.y2
        print(BoundingBox.font())
        text_length = BoundingBox.font().getsize(text)[0]
        draw.rectangle(
            ((x, y), (x+25+text_length+25, y+35)), fill='white')
        draw.rectangle(
            ((x+5, y+5), (x+5+25, y+5+25)), fill=f"#{swatch_color}")
        draw.text(
            (x+5+25+10, y+5), text, fill=text_color, font=BoundingBox.font())

    def move(self, x, y):
        return BoundingBox(
            self.x1 + x,
            self.y1 + y,
            self.x2 + x,
            self.y2 + y
        )

    # Returns new BoundingBox that is square by lengthing the shorter side
    def square(self):
        max_dim = max(self.width, self.height)
        width_diff = max_dim - self.width
        height_diff = max_dim - self.height

        x1_new = self.x1 - width_diff / 2
        x2_new = x1_new + max_dim
        y1_new = self.y1 - height_diff / 2
        y2_new = y1_new + max_dim

        return BoundingBox(x1_new, y1_new, x2_new, y2_new)


    def hash(self):
        bytes = [self.x1, self.y1, self.x2, self.y2].join(".").encode('utf-8')
        return hashlib.sha256(bytes).hexdigest()[:6]

    def __repr__(self):
        return f"BoundingBox({self.x1}, {self.y1}, {self.x2}, {self.y2})"
