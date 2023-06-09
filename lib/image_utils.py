import os
import base64
from io import BytesIO
from PIL import Image, ExifTags
import hashlib
import io

def image_to_data_url(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_data = buffer.getvalue()
    base64_img_data = base64.b64encode(img_data).decode('utf-8')
    return f"data:image/png;base64,{base64_img_data}"

def correct_image_orientation(image):
    try:
        # Get the Exif metadata from the image
        exif_data = image._getexif()

        # Find the orientation tag in the Exif data
        for tag, value in ExifTags.TAGS.items():
            if value == "Orientation":
                orientation_tag = tag
                break

        # Get the orientation value
        orientation = exif_data.get(orientation_tag)

        # Rotate the image based on the orientation value
        if orientation == 3:  # Upside down (180 degrees)
            image = image.rotate(180, resample=Image.BICUBIC, expand=True)
        elif orientation == 6:  # 90 degrees clockwise
            image = image.rotate(270, resample=Image.BICUBIC, expand=True)
        elif orientation == 8:  # 90 degrees counterclockwise
            image = image.rotate(90, resample=Image.BICUBIC, expand=True)

    except (AttributeError, KeyError, IndexError):
        # In case of issues with Exif data, return the original image
        pass

    return image


def get_default_font():
    if os.name == 'nt':  # Windows
        return 'arial.ttf'
    elif os.name == 'posix':  # Linux or Mac
        return '/Library/Fonts/Arial.ttf'

def compute_image_hash(image):
    # Convert the image to bytes
    byte_array = io.BytesIO()
    image.save(byte_array, format='PNG')

    # Compute hash using SHA-256
    hash_object = hashlib.sha256(byte_array.getvalue())
    return hash_object.hexdigest()
