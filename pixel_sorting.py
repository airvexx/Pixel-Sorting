from PIL import Image
import numpy as np
import colorsys
import os

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_PATH       = r"D:\Pictures\DSCF2385.JPG"
OUTPUT_PATH      = r"D:\Pictures\PixelSorting\pixel_sorted_01.jpg"
MASK_PATH        = None        # grayscale mask image (white=sort, black=skip), or None

BATCH_MODE       = False       # set True to process multiple images
BATCH_JOBS       = [           # list of (input_path, output_path) pairs
    # (r"D:\Pictures\img1.jpg", r"D:\Pictures\PixelSorting\img1_sorted.jpg"),
]
BATCH_INPUT_DIR  = None        # process all images in this folder (alternative to BATCH_JOBS)
BATCH_OUTPUT_DIR = None        # save batch results to this folder
BATCH_MASK_PATH  = None        # shared mask applied to all batch images, or None

SORT_AXIS        = "horizontal"  # "horizontal" | "vertical"
SORT_KEY         = "brightness"  # "brightness" | "hue" | "saturation" | "red" | "green" | "blue"
THRESHOLD_MIN    = 23            # pixels with key value below this are not sorted
THRESHOLD_MAX    = 210           # pixels with key value above this are not sorted
REVERSE          = False         # False = low → high; True = high → low
ANGLE            = 0             # rotate image before sorting, then rotate back (degrees)
MASK_THRESHOLD   = 128           # mask pixel value >= this is treated as a sortable region
# ──────────────────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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


def sort_strip(strip, mask_strip=None):
    x = 0
    length = len(strip)
    while x < length:
        val = pixel_key(strip[x])
        in_mask = mask_strip is None or int(mask_strip[x]) >= MASK_THRESHOLD
        if in_mask and THRESHOLD_MIN <= val <= THRESHOLD_MAX:
            start = x
            while x < length:
                v = pixel_key(strip[x])
                m = mask_strip is None or int(mask_strip[x]) >= MASK_THRESHOLD
                if not (m and THRESHOLD_MIN <= v <= THRESHOLD_MAX):
                    break
                x += 1
            strip[start:x] = sorted(strip[start:x], key=pixel_key, reverse=REVERSE)
        else:
            x += 1
    return strip


def sort_image(arr, mask_arr=None):
    if SORT_AXIS == "horizontal":
        for y in range(arr.shape[0]):
            mask_strip = mask_arr[y] if mask_arr is not None else None
            arr[y] = sort_strip(arr[y], mask_strip)
    elif SORT_AXIS == "vertical":
        for x in range(arr.shape[1]):
            mask_strip = mask_arr[:, x] if mask_arr is not None else None
            arr[:, x] = sort_strip(arr[:, x], mask_strip)
    else:
        raise ValueError(f"Unknown SORT_AXIS: '{SORT_AXIS}'")
    return arr


def load_mask(mask_path, target_size):
    """Load a mask image, convert to grayscale, and resize to target (width, height)."""
    mask = Image.open(mask_path).convert("L")
    if mask.size != target_size:
        mask = mask.resize(target_size, Image.LANCZOS)
    return np.array(mask, dtype=np.uint8)


def process_image(input_path, output_path, mask_path=None):
    img = Image.open(input_path).convert("RGB")

    if ANGLE != 0:
        img = img.rotate(ANGLE, expand=True)

    mask_arr = None
    if mask_path:
        mask_arr = load_mask(mask_path, img.size)
        if ANGLE != 0:
            mask_img = Image.fromarray(mask_arr).rotate(ANGLE, expand=True)
            mask_arr = np.array(mask_img, dtype=np.uint8)

    arr = np.array(img, dtype=np.uint8)
    arr = sort_image(arr, mask_arr)
    img = Image.fromarray(arr)

    if ANGLE != 0:
        img = img.rotate(-ANGLE, expand=True)
        original = Image.open(input_path).convert("RGB")
        left = (img.width - original.width) // 2
        top  = (img.height - original.height) // 2
        img = img.crop((left, top, left + original.width, top + original.height))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img.save(output_path)
    print(f"Saved {output_path}")


def build_batch_jobs():
    jobs = list(BATCH_JOBS)
    if BATCH_INPUT_DIR and BATCH_OUTPUT_DIR:
        os.makedirs(BATCH_OUTPUT_DIR, exist_ok=True)
        for fname in sorted(os.listdir(BATCH_INPUT_DIR)):
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                inp = os.path.join(BATCH_INPUT_DIR, fname)
                out = os.path.join(BATCH_OUTPUT_DIR, fname)
                jobs.append((inp, out))
    return jobs


if BATCH_MODE:
    jobs = build_batch_jobs()
    if not jobs:
        print("BATCH_MODE is True but no jobs found. "
              "Populate BATCH_JOBS or set BATCH_INPUT_DIR / BATCH_OUTPUT_DIR.")
    for i, (inp, out) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {inp} ...")
        process_image(inp, out, mask_path=BATCH_MASK_PATH)
else:
    process_image(INPUT_PATH, OUTPUT_PATH, mask_path=MASK_PATH)