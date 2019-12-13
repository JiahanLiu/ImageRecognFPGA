import sys
import numpy as np
from PIL import Image

if len(sys.argv) <= 1:
    print("Usage: python create_test_image.py <path to JPG>")
    exit(0)


imgpath = sys.argv[1]

with Image.open(imgpath) as imgPIL:
    img = np.asarray(imgPIL)

with open("test_image.c", "w") as f:
    f.write("#include <stdint.h>\n")
    f.write("uint8_t test_image[] = {\n")
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            for ch in range(img.shape[2]):
                if not (r==0 and c==0 and ch==0):
                    f.write(", ")
                f.write(str(img[r,c,ch]))
    f.write("};\n")

