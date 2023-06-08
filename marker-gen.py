import os
import cv2
from cv2 import aruco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lib.lego_colors import lego_colors_by_id
from lib.image_utils import get_default_font

dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
small_font = ImageFont.truetype(get_default_font(), 50)
large_font = ImageFont.truetype(get_default_font(), 175)

SymbolSize = 300
BorderSize = 100
SymbolWithBorderSize = SymbolSize + (BorderSize*2)
TextWidth = 625
ColorX = SymbolSize*2 + BorderSize*3
ColorWidth = BorderSize
TextX = ColorX + BorderSize + 25
CardWidth = SymbolSize*2 + BorderSize*3 + TextWidth
CardHeight = SymbolWithBorderSize


card_images = []

color_ids = [
    0,
    1,
    10,
    1050,
    14,
    15,
    158,
    182,
    19,
    191,
    2,
    212,
    226,
    25,
    26,
    27,
    272,
    28,
    288,
    29,
    3,
    30,
    308,
    31,
    320,
    321,
    322,
    323,
    326,
    378,
    4,
    41,
    46,
    47,
    484,
    5,
    70,
    71,
    72,
    73,
    84,
    85,
]
for color_id in sorted(color_ids):
    color = lego_colors_by_id[color_id]

    # We are using ArUco markers with a 100 code dictionary so they can represent 0 - 99.
    # To represent a Rebrickable ID (0 - 1095) we use two symbols:
	#   * first symbol will be in the range 0-49 and represents 0-49
	#   * second symbol will be in the range 51-90 and represents (x - 50) * 50
    # So 226 would be represented by symbols 26 and 4 (26 * 4 * 50 = 226)
    symbol1_id = color.id % 50
    symbol2_id = 50 + (color.id // 50)
    print(f"{color.id} = {symbol1_id} + {symbol2_id}")

    symbol1_array = aruco.generateImageMarker(dict, symbol1_id, SymbolSize)
    symbol2_array = aruco.generateImageMarker(dict, symbol2_id, SymbolSize)
    symbol1 = Image.fromarray(cv2.cvtColor(symbol1_array, cv2.COLOR_BGR2RGB))
    symbol2 = Image.fromarray(cv2.cvtColor(symbol2_array, cv2.COLOR_BGR2RGB))

    image = Image.new('RGBA', (CardWidth, CardHeight), 'white')
    image.paste(symbol1, (BorderSize, BorderSize))
    image.paste(symbol2, (SymbolSize + BorderSize*2, BorderSize))
    d = ImageDraw.Draw(image)

    d.rectangle((ColorX, BorderSize, ColorX + ColorWidth, CardHeight-BorderSize), fill=tuple(color.rgb()), width=1)

    d.text((TextX, BorderSize+40), str(color.id), fill="black", font=large_font)
    d.text((TextX, BorderSize+240), color.name, fill="black", font=small_font)

    d.rectangle((0, 0, image.width-1, image.height-1), outline='LightGray', width=1)

    card_images.append(image)

    image.save("tmp/t.png")


NumColumns = 2
BatchSize = 16

for batch_index in range(0, len(card_images), BatchSize):
    print(f"Generating sheet {batch_index} - {batch_index+BatchSize}")
    batch_of_card_images = card_images[batch_index:batch_index+BatchSize]

    NumRows = (len(batch_of_card_images) + NumColumns - 1) // NumColumns
    OutputWidth = NumColumns * CardWidth
    OutputHeight = NumRows * CardHeight

    output_image = Image.new('RGB', (OutputWidth, OutputHeight), 'white')

    for i, image in enumerate(batch_of_card_images):
        row = i // NumColumns
        col = i % NumColumns
        x = col * CardWidth
        y = row * CardHeight
        output_image.paste(image, (x, y))

    # Save the output image
    print(f"Saving tmp/marker-sheet-{batch_index}.png")
    output_image.save(f"tmp/marker-sheet-{batch_index}.png")
