import time
import re
import numpy as np
from PIL import Image
from pathlib import Path
from numba import njit

@njit
def read_ppm_file(contents, h, w, pixels):
    h, w = int(h), int(w)
    pixel_array = np.empty((h, w, 3), dtype=np.uint8)

    pixel_idx = 0
    for i in range(h):
        for j in range(w):
            for k in range(3):
                pixel_array[i, j, k] = pixels[pixel_idx]
                pixel_idx += 1

    return pixel_array

def save_as_image(pixel_array, output_path="out.jpg"):
    Image.fromarray(pixel_array).save(output_path)

def main():
    start = time.time()
    ppm_file_path = 'out.ppm'
    contents = Path(ppm_file_path).read_text()
    identifier, h, w, *pixels = map(int, re.findall(r'[0-9]+', contents))
    
    pixels_array = read_ppm_file(contents, h, w, pixels)
    save_as_image(pixels_array)
    end = time.time()
    print(end - start)

if __name__ == "__main__":
    main()
