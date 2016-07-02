# script to extract rgb values from kuler strips :)

from PIL import Image
import sys
from string import Template

BLOCK_ELEMS = [21, 21, 20, 21, 21]
BLOCK_HEIGHT = 21
BLOCK_SIZE = sum(BLOCK_ELEMS)

SPACING_X = 13
SPACING_Y = 21

im = Image.open(sys.argv[1])
w, h = im.size
print w, h

BLOCKS_U8 = []
BLOCKS_FLOAT = []

y = 0
while y < h:
    x = 0
    while x < w:
        # process a block
        block = []
        x_inner = x
        for i in range(5):
            r, g, b = im.getpixel((x_inner, y))
            x_inner += BLOCK_ELEMS[i]
            block.extend((r, g, b))

        rr = all([block[i] == block[i+1] for i in range(len(block)-1)])
        if not rr:
            BLOCKS_U8.append(block)
            BLOCKS_FLOAT.append(map(lambda x: "%.3f" % (x / 255.0), block))

        x += BLOCK_SIZE + SPACING_X
    y += BLOCK_HEIGHT + SPACING_Y

# each element represents 5 rgb triplets

template_u8 = """
u8 cols_u8[] = {
$vals
};
"""

template_float = """
float cols_float[] = {
$vals
};
"""

rows_u8 = [', '.join(map(str, x)) for x in BLOCKS_U8]
rows_u8 = ['  ' + x for x in rows_u8]
vals_u8 = ',\n'.join(rows_u8)
print Template(template_u8).substitute(vals=vals_u8)


rows_float = [', '.join(map(str, x)) for x in BLOCKS_FLOAT]
rows_float = ['  ' + x for x in rows_float]
vals_float = ',\n'.join(rows_float)
print Template(template_float).substitute(vals=vals_float)

