# stack_peloton.py
# pip install pillow
from PIL import Image, ImageColor
from datetime import datetime

# === set your three images in the order you want ===
paths = [
    r"20251106_2113_ac38e9a2_stacked.png",  # Trance
    r"20251106_2144_cc09a3a6_stacked.png",  # Pop Punk
    r"20251106_2222_2bcb5a85_stacked.png",  # Sample That
]

# spacer look
SPACER_HEIGHT = 32                    # px
SPACER_COLOR  = "#111216"             # dark gray/near-black

# --- load ---
imgs = [Image.open(p).convert("RGBA") for p in paths]

# --- normalize widths to the widest image ---
max_w = max(im.width for im in imgs)
norm = []
for im in imgs:
    if im.width != max_w:
        ratio = max_w / im.width
        im = im.resize((max_w, int(im.height * ratio)), Image.LANCZOS)
    norm.append(im)

# --- spacer ---
spacer_rgba = ImageColor.getrgb(SPACER_COLOR) + (255,)
spacer = Image.new("RGBA", (max_w, SPACER_HEIGHT), spacer_rgba)

# --- stack: img1 + spacer + img2 + spacer + img3 ---
layers = []
for i, im in enumerate(norm):
    layers.append(im)
    if i < len(norm) - 1:
        layers.append(spacer)

total_h = sum(im.height for im in layers)
combined = Image.new("RGBA", (max_w, total_h), (0, 0, 0, 0))

y = 0
for im in layers:
    combined.paste(im, (0, y))
    y += im.height

out_path = f"peloton_combo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
combined.save(out_path)
print("Saved:", out_path)