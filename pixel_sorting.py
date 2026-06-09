from PIL import Image, ImageTk
import numpy as np
import colorsys
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_PATH = r"D:\Pictures\DSCF2385.JPG"
OUTPUT_PATH = r"D:\Pictures\PixelSorting\pixel_sorted_02.jpg"
MASK_PATH = None  # grayscale mask image (white=sort, black=skip), or None

BATCH_MODE = False  # set True to process multiple images
BATCH_JOBS = [  # list of (input_path, output_path) pairs
    # (r"D:\Pictures\img1.jpg", r"D:\Pictures\PixelSorting\img1_sorted.jpg"),
]
BATCH_INPUT_DIR = None  # process all images in this folder (alternative to BATCH_JOBS)
BATCH_OUTPUT_DIR = None  # save batch results to this folder
BATCH_MASK_PATH = None  # shared mask applied to all batch images, or None

SORT_AXIS = "horizontal"  # "horizontal" | "vertical"
SORT_KEY = (
    "brightness"  # "brightness" | "hue" | "saturation" | "red" | "green" | "blue"
)
THRESHOLD_MIN = 23  # pixels with key value below this are not sorted
THRESHOLD_MAX = 210  # pixels with key value above this are not sorted
REVERSE = False  # False = low → high; True = high → low
ANGLE = 0  # rotate image before sorting, then rotate back (degrees)
MASK_THRESHOLD = 128  # mask pixel value >= this is treated as a sortable region
LAUNCH_GUI = True  # True = open control panel instead of running immediately
# ──────────────────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SORT_AXIS_OPTIONS = ["horizontal", "vertical"]
SORT_KEY_OPTIONS = ["brightness", "hue", "saturation", "red", "green", "blue"]
PREVIEW_MAX_SIZE = 360


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
        top = (img.height - original.height) // 2
        img = img.crop((left, top, left + original.width, top + original.height))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img.save(output_path)
    print(f"Saved {output_path}")


def render_preview_image(input_path, mask_path=None):
    img = Image.open(input_path).convert("RGB")
    img.thumbnail((PREVIEW_MAX_SIZE, PREVIEW_MAX_SIZE), Image.LANCZOS)
    original = img.copy()

    if ANGLE != 0:
        img = img.rotate(ANGLE, expand=True)

    mask_arr = None
    if mask_path:
        mask_arr = load_mask(mask_path, img.size)

    arr = np.array(img, dtype=np.uint8)
    arr = sort_image(arr, mask_arr)
    img = Image.fromarray(arr)

    if ANGLE != 0:
        img = img.rotate(-ANGLE, expand=True)
        left = (img.width - original.width) // 2
        top = (img.height - original.height) // 2
        img = img.crop((left, top, left + original.width, top + original.height))

    return img


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


def run_with_current_settings():
    if BATCH_MODE:
        jobs = build_batch_jobs()
        if not jobs:
            print(
                "BATCH_MODE is True but no jobs found. "
                "Populate BATCH_JOBS or set BATCH_INPUT_DIR / BATCH_OUTPUT_DIR."
            )
            return
        for i, (inp, out) in enumerate(jobs, 1):
            print(f"[{i}/{len(jobs)}] {inp} ...")
            process_image(inp, out, mask_path=BATCH_MASK_PATH)
    else:
        process_image(INPUT_PATH, OUTPUT_PATH, mask_path=MASK_PATH)


def launch_gui():
    root = tk.Tk()
    root.title("Pixel Sorting")
    root.resizable(False, False)

    input_var = tk.StringVar(value=INPUT_PATH)
    output_var = tk.StringVar(value=OUTPUT_PATH)
    mask_var = tk.StringVar(value=MASK_PATH or "")
    axis_var = tk.StringVar(value=SORT_AXIS)
    key_var = tk.StringVar(value=SORT_KEY)
    thresh_min_var = tk.DoubleVar(value=THRESHOLD_MIN)
    thresh_max_var = tk.DoubleVar(value=THRESHOLD_MAX)
    reverse_var = tk.BooleanVar(value=REVERSE)
    angle_var = tk.DoubleVar(value=ANGLE)
    mask_threshold_var = tk.IntVar(value=MASK_THRESHOLD)
    preview_status_var = tk.StringVar(value="Preview: waiting for input")
    preview_photo = {"img": None}
    preview_job = {"id": None}

    frame = ttk.Frame(root, padding=12)
    frame.grid(row=0, column=0, sticky="nsew")

    ttk.Label(frame, text="Input Image").grid(row=0, column=0, sticky="w")
    ttk.Entry(frame, textvariable=input_var, width=54).grid(row=1, column=0, sticky="we")
    ttk.Button(
        frame,
        text="Browse",
        command=lambda: input_var.set(
            filedialog.askopenfilename(
                title="Select Input Image",
                filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")],
            )
            or input_var.get()
        ),
    ).grid(row=1, column=1, padx=(8, 0))

    ttk.Label(frame, text="Output Image").grid(row=2, column=0, sticky="w", pady=(10, 0))
    ttk.Entry(frame, textvariable=output_var, width=54).grid(row=3, column=0, sticky="we")
    ttk.Button(
        frame,
        text="Browse",
        command=lambda: output_var.set(
            filedialog.asksaveasfilename(
                title="Save Output Image",
                defaultextension=".jpg",
                filetypes=[
                    ("JPEG", "*.jpg"),
                    ("PNG", "*.png"),
                    ("TIFF", "*.tif"),
                    ("WebP", "*.webp"),
                    ("All files", "*.*"),
                ],
            )
            or output_var.get()
        ),
    ).grid(row=3, column=1, padx=(8, 0))

    ttk.Label(frame, text="Mask Image (Optional)").grid(row=4, column=0, sticky="w", pady=(10, 0))
    ttk.Entry(frame, textvariable=mask_var, width=54).grid(row=5, column=0, sticky="we")
    mask_buttons = ttk.Frame(frame)
    mask_buttons.grid(row=5, column=1, padx=(8, 0), sticky="n")
    ttk.Button(
        mask_buttons,
        text="Browse",
        command=lambda: mask_var.set(
            filedialog.askopenfilename(
                title="Select Mask Image",
                filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")],
            )
            or mask_var.get()
        ),
    ).grid(row=0, column=0, sticky="we")
    ttk.Button(mask_buttons, text="Clear", command=lambda: mask_var.set("")).grid(
        row=1, column=0, pady=(6, 0), sticky="we"
    )

    options = ttk.LabelFrame(frame, text="Glitch Parameters", padding=10)
    options.grid(row=6, column=0, columnspan=2, pady=(12, 0), sticky="we")

    ttk.Label(options, text="Sort Axis").grid(row=0, column=0, sticky="w")
    ttk.Combobox(
        options,
        textvariable=axis_var,
        values=SORT_AXIS_OPTIONS,
        state="readonly",
        width=16,
    ).grid(row=1, column=0, sticky="w", padx=(0, 12))

    ttk.Label(options, text="Sort Key").grid(row=0, column=1, sticky="w")
    ttk.Combobox(
        options,
        textvariable=key_var,
        values=SORT_KEY_OPTIONS,
        state="readonly",
        width=16,
    ).grid(row=1, column=1, sticky="w", padx=(0, 12))

    ttk.Label(options, text="Threshold Min").grid(row=0, column=2, sticky="w")
    ttk.Entry(options, textvariable=thresh_min_var, width=10).grid(row=1, column=2, sticky="w")

    ttk.Label(options, text="Threshold Max").grid(row=2, column=0, sticky="w", pady=(10, 0))
    ttk.Entry(options, textvariable=thresh_max_var, width=10).grid(row=3, column=0, sticky="w")

    ttk.Label(options, text="Angle").grid(row=2, column=1, sticky="w", pady=(10, 0))
    ttk.Entry(options, textvariable=angle_var, width=10).grid(row=3, column=1, sticky="w")

    ttk.Label(options, text="Mask Threshold").grid(row=2, column=2, sticky="w", pady=(10, 0))
    ttk.Entry(options, textvariable=mask_threshold_var, width=10).grid(row=3, column=2, sticky="w")

    ttk.Checkbutton(options, text="Reverse Sort", variable=reverse_var).grid(
        row=4, column=0, columnspan=2, sticky="w", pady=(10, 0)
    )

    preview_box = ttk.LabelFrame(frame, text="Live Preview", padding=10)
    preview_box.grid(row=7, column=0, columnspan=2, pady=(12, 0), sticky="we")
    preview_label = ttk.Label(preview_box, text="No preview yet", width=48, anchor="center")
    preview_label.grid(row=0, column=0, sticky="we")
    ttk.Label(preview_box, textvariable=preview_status_var).grid(row=1, column=0, sticky="w", pady=(8, 0))

    def apply_settings_from_ui(require_output=False):
        global INPUT_PATH, OUTPUT_PATH, MASK_PATH
        global SORT_AXIS, SORT_KEY, THRESHOLD_MIN, THRESHOLD_MAX, REVERSE, ANGLE, MASK_THRESHOLD

        inp = input_var.get().strip()
        out = output_var.get().strip()
        mask = mask_var.get().strip() or None

        if not inp:
            raise ValueError("Input image path is required.")
        if not os.path.isfile(inp):
            raise ValueError(f"Input image not found: {inp}")
        if require_output and not out:
            raise ValueError("Output image path is required.")
        if mask and not os.path.isfile(mask):
            raise ValueError(f"Mask image not found: {mask}")

        sort_axis = axis_var.get()
        sort_key = key_var.get()
        threshold_min = float(thresh_min_var.get())
        threshold_max = float(thresh_max_var.get())
        reverse = bool(reverse_var.get())
        angle = float(angle_var.get())
        mask_threshold = int(mask_threshold_var.get())

        if threshold_min > threshold_max:
            raise ValueError("Threshold Min must be <= Threshold Max.")
        if not (0 <= mask_threshold <= 255):
            raise ValueError("Mask Threshold must be between 0 and 255.")

        INPUT_PATH = inp
        OUTPUT_PATH = out
        MASK_PATH = mask
        SORT_AXIS = sort_axis
        SORT_KEY = sort_key
        THRESHOLD_MIN = threshold_min
        THRESHOLD_MAX = threshold_max
        REVERSE = reverse
        ANGLE = angle
        MASK_THRESHOLD = mask_threshold

    def update_preview():
        preview_job["id"] = None
        try:
            apply_settings_from_ui(require_output=False)
            img = render_preview_image(INPUT_PATH, mask_path=MASK_PATH)
            tk_img = ImageTk.PhotoImage(img)
            preview_label.configure(image=tk_img, text="")
            preview_photo["img"] = tk_img
            preview_status_var.set(f"Preview: {img.width}x{img.height}")
        except Exception as exc:
            preview_label.configure(image="", text="Preview unavailable")
            preview_photo["img"] = None
            preview_status_var.set(f"Preview: {exc}")

    def schedule_preview(*_):
        if preview_job["id"] is not None:
            root.after_cancel(preview_job["id"])
        preview_job["id"] = root.after(250, update_preview)

    def run_from_gui():
        try:
            apply_settings_from_ui(require_output=True)
            process_image(INPUT_PATH, OUTPUT_PATH, mask_path=MASK_PATH)
            messagebox.showinfo("Pixel Sorting", f"Saved output to:\n{OUTPUT_PATH}")
        except Exception as exc:
            messagebox.showerror("Pixel Sorting", str(exc))

    button_bar = ttk.Frame(frame)
    button_bar.grid(row=8, column=0, columnspan=2, pady=(12, 0), sticky="we")
    ttk.Button(button_bar, text="Refresh Preview", command=update_preview).grid(row=0, column=0, padx=(0, 8))
    ttk.Button(button_bar, text="Run Pixel Sort", command=run_from_gui).grid(row=0, column=1, sticky="we")

    for var in (
        input_var,
        output_var,
        mask_var,
        axis_var,
        key_var,
        thresh_min_var,
        thresh_max_var,
        reverse_var,
        angle_var,
        mask_threshold_var,
    ):
        var.trace_add("write", schedule_preview)

    schedule_preview()

    root.mainloop()


if __name__ == "__main__":
    if LAUNCH_GUI:
        launch_gui()
    else:
        run_with_current_settings()
