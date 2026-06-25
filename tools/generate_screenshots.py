"""
Generate professional terminal screenshots from ANSI-colored output.
Saves PNG images that look like real terminal screenshots.
"""

import re, os
from PIL import Image, ImageDraw, ImageFont

TEMP = os.environ.get("TEMP", ".")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "screenshots")
os.makedirs(OUT_DIR, exist_ok=True)

FONT_PATH = "C:/Windows/Fonts/Consolas.ttf"
FONT_SIZE = 14
LINE_HEIGHT = 20
CHAR_WIDTH = 8
PADDING_X = 24
PADDING_Y = 16
HEADER_HEIGHT = 32

ANSI_COLORS = {
    "30": (0, 0, 0),
    "31": (200, 40, 40),
    "32": (60, 200, 80),
    "33": (220, 180, 40),
    "34": (60, 120, 220),
    "35": (200, 80, 200),
    "36": (40, 180, 200),
    "37": (200, 200, 200),
    "90": (100, 100, 100),
    "91": (255, 80, 80),
    "92": (80, 255, 100),
    "93": (255, 220, 60),
    "94": (80, 150, 255),
    "95": (255, 100, 255),
    "96": (60, 220, 240),
    "97": (240, 240, 240),
}

BG_COLOR = (18, 18, 18)
HEADER_BG = (30, 30, 30)
BORDER_COLOR = (60, 60, 60)
TITLE_COLOR = (200, 200, 200)
TAB_COLOR = (50, 50, 50)
TAB_ACTIVE_BG = (18, 18, 18)

def strip_ansi(text):
    return re.sub(r"\033\[[0-9;]*m", "", text)

def parse_ansi_lines(text):
    """Parse ANSI text into list of (clean_text, segments_with_colors)."""
    raw_lines = text.split("\n")
    parsed = []
    for raw in raw_lines:
        segments = []
        pos = 0
        current_color = (200, 200, 200)
        bold = False
        dim = False

        for match in re.finditer(r"\033\[([0-9;]*)m", raw):
            if match.start() > pos:
                segments.append((raw[pos:match.start()], current_color, bold, dim))
            code_str = match.group(1)
            if not code_str:
                current_color = (200, 200, 200)
                bold = False
                dim = False
            else:
                codes = code_str.split(";")
                for c in codes:
                    if c == "1":
                        bold = True
                    elif c == "2":
                        dim = True
                    elif c == "0":
                        current_color = (200, 200, 200)
                        bold = False
                        dim = False
                    elif c in ANSI_COLORS:
                        current_color = ANSI_COLORS[c]
                        if dim:
                            current_color = tuple(int(v * 0.5) for v in current_color)
            pos = match.end()

        if pos < len(raw):
            segments.append((raw[pos:], current_color, bold, dim))

        parsed.append((raw, segments))
    return parsed

def truncate_carriage_returns(lines):
    """Handle \r by taking only the last segment before \r or after last \r."""
    result = []
    for raw, segments in lines:
        parts = raw.split("\r")
        last = parts[-1] if parts else ""
        if last and last.strip():
            result.append((last, [(last, (200, 200, 200), False, False)]))
        else:
            result.append((raw, segments))
    return result

def render_terminal_screenshot(title, input_file, output_file, max_width=120):
    with open(input_file, "r", encoding="utf-8", errors="replace") as f:
        raw_text = f.read()

    raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    lines_data = parse_ansi_lines(raw_text)

    cleaned_lines = [
        strip_ansi(raw) if re.search(r"\r", raw) else raw
        for raw, _ in lines_data
    ]

    display_lines = []
    skip_cr = False
    for raw, segments in lines_data:
        if "\r" in raw:
            parts = raw.split("\r")
            last = parts[-1]
            p_segments = parse_ansi_lines(last)[0][1] if last.strip() else segments
            display_lines.append((last, p_segments))
        else:
            display_lines.append((raw, segments))

    final_clean = [strip_ansi(raw) for raw, _ in display_lines]

    last_nonempty = len(final_clean) - 1
    while last_nonempty >= 0 and not final_clean[last_nonempty].strip():
        last_nonempty -= 1

    if last_nonempty >= 0:
        display_lines = display_lines[: last_nonempty + 1]
        final_clean = final_clean[: last_nonempty + 1]

    max_line_len = max((len(l) for l in final_clean), default=80)
    max_line_len = min(max(max_line_len, 80), max_width)

    canvas_w = max_line_len * CHAR_WIDTH + PADDING_X * 2
    canvas_h = len(display_lines) * LINE_HEIGHT + PADDING_Y * 2 + HEADER_HEIGHT

    img = Image.new("RGB", (int(canvas_w), int(canvas_h)), BG_COLOR)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except:
        font = ImageFont.load_default()

    draw.rectangle([(0, 0), (int(canvas_w) - 1, int(canvas_h) - 1)], outline=BORDER_COLOR, width=1)

    draw.rectangle([(1, 1), (int(canvas_w) - 2, HEADER_HEIGHT)], fill=HEADER_BG)

    draw.rectangle([(8, HEADER_HEIGHT - 2), (16, HEADER_HEIGHT - 2)], fill=TAB_COLOR)

    circle_radius = 6
    for cx, col in [(PADDING_X - 8, (255, 95, 87)), (PADDING_X + 4, (255, 189, 46)),
                    (PADDING_X + 16, (39, 201, 63))]:
        draw.ellipse([(cx - circle_radius, 8 - circle_radius),
                      (cx + circle_radius, 8 + circle_radius)], fill=col)

    title_text = f"  {title}"
    draw.text((PADDING_X + 40, 6), title_text, font=font, fill=TITLE_COLOR)

    y = PADDING_Y + HEADER_HEIGHT
    for raw, segments in display_lines:
        if not raw.strip() and segments and all(not s[0].strip() for s in segments):
            y += LINE_HEIGHT
            continue

        x_cursor = PADDING_X
        if segments:
            for text, color, bold, dim in segments:
                if not text:
                    continue
                try:
                    c = tuple(int(v) for v in color)
                except:
                    c = (200, 200, 200)
                draw.text((x_cursor, y), text, font=font, fill=c)
                x_cursor += len(text) * CHAR_WIDTH
        else:
            draw.text((x_cursor, y), raw, font=font, fill=(200, 200, 200))
        y += LINE_HEIGHT

        if y > int(canvas_h) - PADDING_Y:
            break

    img.save(output_file, "PNG")
    return output_file


for fname, title, mw in [
    ("train_output.txt", "Spam Detector v3 — Training Mode (98.18% Accuracy)", 100),
    ("demo_output.txt",  "Spam Detector v3 — Full Demo (All Features)", 130),
    ("app_output.txt",   "Spam Detector v3 — Interactive App", 100),
]:
    fpath = os.path.join(TEMP, fname)
    if os.path.exists(fpath):
        out = render_terminal_screenshot(title, fpath,
                                         os.path.join(OUT_DIR, f"screenshot_{fname.replace('_output.txt', '')}.png"),
                                         max_width=mw)
        print(f"Created: {out}")

print(f"\nAll screenshots saved to: {OUT_DIR}")
