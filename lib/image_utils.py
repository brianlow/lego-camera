import base64
from io import BytesIO

def image_to_data_url(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_data = buffer.getvalue()
    base64_img_data = base64.b64encode(img_data).decode('utf-8')
    return f"data:image/png;base64,{base64_img_data}"
