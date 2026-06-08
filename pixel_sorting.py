from PIL import Image
import numpy as np
import colorsys

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_PATH     = "DSCF2625.jpg"
OUTPUT_PATH    = "pixel_sorted_11.jpg"

SORT_AXIS      = "horizontal"  # "horizontal" | "vertical"
SORT_KEY       = "brightness"  # "brightness" | "hue" | "saturation" | "red" | "green" | "blue"
THRESHOLD_MIN  = 23            # pixels with key value below this are not sorted
THRESHOLD_MAX  = 890           # pixels with key value above this are not sorted
REVERSE        = False         # False = low → high; True = high → low
ANGLE          = 0             # rotate image before sorting, then rotate back (degrees)
# ──────────────────────────────────────────────────────────────────────────────


def pixel_key(pixel):
    r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
    if SORT_KEY == "brightness":
        return 0.299 * r + 0.587 * g + 0.114 * b
    elif SORT_KEY == "hue":
        h, _, _ = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        return h * 360
    elif SORT_KEY == "saturation":
        _, s, _ = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        return s * 255
    elif SORT_KEY == "red":
        return r
    elif SORT_KEY == "green":
        return g
    elif SORT_KEY == "blue":
        return b
    else:
        raise ValueError(f"Unknown SORT_KEY: '{SORT_KEY}'")


def sort_strip(strip):
    x = 0
    length = len(strip)
    while x < length:
        val = pixel_key(strip[x])
        if THRESHOLD_MIN <= val <= THRESHOLD_MAX:
            start = x
            while x < length and THRESHOLD_MIN <= pixel_key(strip[x]) <= THRESHOLD_MAX:
                x += 1
            segment = strip[start:x]
            strip[start:x] = sorted(segment, key=pixel_key, reverse=REVERSE)
        else:
            x += 1
    return strip


def sort_image(arr):
    if SORT_AXIS == "horizontal":
        for y in range(arr.shape[0]):
            arr[y] = sort_strip(arr[y])
    elif SORT_AXIS == "vertical":
        for x in range(arr.shape[1]):
            arr[:, x] = sort_strip(arr[:, x])
    else:
        raise ValueError(f"Unknown SORT_AXIS: '{SORT_AXIS}'")
    return arr


img = Image.open(INPUT_PATH).convert("RGB")

if ANGLE != 0:
    img = img.rotate(ANGLE, expand=True)

arr = np.array(img, dtype=np.uint8)
arr = sort_image(arr)
img = Image.fromarray(arr)

if ANGLE != 0:
    img = img.rotate(-ANGLE, expand=True)
    original = Image.open(INPUT_PATH).convert("RGB")
    left = (img.width - original.width) // 2
    top = (img.height - original.height) // 2
    img = img.crop((left, top, left + original.width, top + original.height))

img.save(OUTPUT_PATH)
print(f"Saved {OUTPUT_PATH}")