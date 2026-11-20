#!/usr/bin/env python3
"""
Builds auto_executive_test_inventory.json from the source dealer's plain
"legacy" inventory pages, and saves the car photos into this repo so the
public feed never points back to the source site.

1. Reads the full list page and collects every car link.
2. Opens each car page for the title, mileage and photo URLs.
3. Downloads the photos into photos/<car id>/ (resized, a few per car).
4. Writes the JSON file your Squarespace page reads. Photo links in it
   point only to your own GitHub repo.

Safety: if far fewer cars than expected are found (for example the source
site blocks us), the existing JSON file is left untouched and the run fails.
"""

import hashlib
import io
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError:  # photos are still saved, just not resized
    Image = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from scipy import ndimage
except ImportError:
    ndimage = None

try:
    from ultralytics import YOLO       # finds each car in the photo (optional)
except Exception:
    YOLO = None

try:
    from rembg import new_session, remove as rembg_remove
except ImportError:
    rembg_remove = None
    new_session = None

# ---- Settings you may want to change ---------------------------------------
BASE = "https://sunshineautobrokers.com"
LIST_URL = BASE + "/legacy"
OUTPUT = "auto_executive_test_inventory.json"
PHOTO_DIR = Path("photos")
PUBLIC_PHOTO_BASE = (
    "https://raw.githubusercontent.com/AutoExecutive26/"
    "auto-executive-public-feed/main/photos"
)
MIN_EXPECTED = 30          # refuse to overwrite the feed with fewer cars than this
MAX_PHOTOS = 5             # photos saved per car (keeps the repo small)
MAX_WIDTH = 1000           # photos are shrunk to this width in pixels
MAX_DOWNLOAD_BYTES = 8_000_000
# Backdrop for every photo. Choose one:
#   "blackgold" = gold glow in the middle fading to black at the edges
#   "gold"      = solid gold look
#   "none"      = keep the original photo scenery (not recommended)
BACKGROUND_STYLE = "blackgold"
CANVAS_W, CANVAS_H = 1000, 562       # matches the 16:9 cards on your page
# A cutout is only used if the car fills between these shares of the photo.
# This drops interior shots, close-ups and failed cutouts.
MIN_CAR_SHARE, MAX_CAR_SHARE = 0.12, 0.75
STYLE_VERSION = "v6"                 # bump to force every photo to be redone
MAX_CANDIDATES = 8         # photos tried per car; the cleanest ones are kept
MIN_SCORE = 40             # photos scoring below this (0-99) are dropped
DETECTOR_MODEL = "yolov8m-seg.pt"   # finds each separate car in a photo
GATE_PX = 14               # how far outside the detected car the cutout may reach
MASK_ALPHA = 160           # how solid a pixel must be to count as part of the car
ERODE_PX = 9               # used to separate the main car from things touching it
# Dealer logo / watermark area of the ORIGINAL photo, as fractions of its size:
# (left, top, right, bottom). Everything inside is erased before the cutout.
# If a photo's car reaches far into this box, the photo is skipped instead.
WATERMARK_BOX = None          # OFF: erasing a big box was cutting chunks out of cars
MAX_WATERMARK_OVERLAP = 0.12
MAX_REMOVED_SHARE = 0.30   # more than this removed = several cars in frame, skip
EDGE_PX = 2                # car touching the left/right edge = cut off, skip
MIN_RATIO, MAX_RATIO = 1.3, 4.8   # width/height of a normal exterior car shot
PHOTO_CACHE = Path("photo_cache.json")        # remembers photos already rejected
BLOCKLIST_FILE = Path("photo_blocklist.txt")  # photos you never want shown
DELAY_SECONDS = 0.6        # pause between requests, to be polite to the source
# Words that must never appear in titles on your site (case-insensitive)
BANNED_WORDS = ["sunshine auto", "sunshine"]
# -----------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

TWO_WORD_MAKES = ["Land Rover", "Alfa Romeo", "Aston Martin", "Rolls Royce"]
SKIP_IMG = re.compile(
    r"logo|icon|sprite|favicon|placeholder|no-?image|spacer|pixel|"
    r"facebook|twitter|instagram|badge|banner|carfax|autocheck",
    re.I,
)
IMG_EXT = re.compile(r"\.(?:jpe?g|png|webp)(?:\?|$)", re.I)


def fetch(session, url, tries=3):
    for attempt in range(1, tries + 1):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.text
            print(f"  HTTP {r.status_code} for {url} (try {attempt})")
        except requests.RequestException as exc:
            print(f"  Error for {url}: {exc} (try {attempt})")
        time.sleep(2 * attempt)
    return None


def list_cars(list_html):
    soup = BeautifulSoup(list_html, "html.parser")
    cars = {}
    for a in soup.find_all("a", href=True):
        m = re.search(r"/legacy/view/(\d+)", a["href"])
        if m and m.group(1) not in cars:
            cars[m.group(1)] = urljoin(LIST_URL, a["href"])
    return cars


def scrub(text):
    """Remove any banned words (like the source dealer's name) from text."""
    for word in BANNED_WORDS:
        text = re.sub(re.escape(word), "", text, flags=re.I)
    return " ".join(text.split())


def clean_title(title):
    words = []
    for w in scrub(title).split():
        if w.isalpha() and w.isupper() and len(w) > 3:
            w = w.capitalize()
        words.append(w)
    return " ".join(words)


def find_title(soup):
    for tag in soup.find_all(["h1", "h2", "h3", "h4"]):
        text = " ".join(tag.get_text(" ", strip=True).split())
        if re.match(r"^(19|20)\d{2}\s+\S+", text):
            return clean_title(text)
    return None


def split_title(title):
    m = re.match(r"^((?:19|20)\d{2})\s+(.*)$", title)
    if not m:
        return None, "", title
    year = int(m.group(1))
    rest = m.group(2)
    make = None
    for two in TWO_WORD_MAKES:
        if rest.lower().startswith(two.lower() + " ") or rest.lower() == two.lower():
            make = rest[: len(two)]
            break
    if make is None:
        make = rest.split(" ", 1)[0]
    model = rest[len(make):].strip()
    return year, make, model


def find_mileage(page_text):
    patterns = [
        r"(?:mileage|odometer|miles)\s*[:\-]?\s*([\d,]{3,9})\b",
        r"\b([\d,]{4,9})\s*(?:miles|mi\b)",
    ]
    for pat in patterns:
        m = re.search(pat, page_text, re.I)
        if m:
            try:
                value = int(m.group(1).replace(",", ""))
            except ValueError:
                continue
            if 0 < value < 999999:
                return value
    return None


def find_photos(soup, page_url, raw_html):
    found = []

    def add(url):
        if not url:
            return
        url = url.strip()
        if url.startswith(("data:", "javascript:", "#")):
            return
        url = urljoin(page_url, url)
        if url.startswith("http://"):
            url = "https://" + url[len("http://"):]
        if not url.startswith("https://"):
            return
        if SKIP_IMG.search(url) or re.search(r"\.(?:svg|gif)(?:\?|$)", url, re.I):
            return
        if url not in found:
            found.append(url)

    for meta in soup.find_all("meta", attrs={"property": "og:image"}):
        add(meta.get("content"))

    for img in soup.find_all("img"):
        for attr in ("src", "data-src", "data-lazy-src", "data-original"):
            add(img.get(attr))
        srcset = img.get("data-srcset") or img.get("srcset")
        if srcset:
            add(srcset.split(",")[0].strip().split(" ")[0])

    for a in soup.find_all("a", href=True):
        if IMG_EXT.search(a["href"]):
            add(a["href"])

    # Image URLs tucked inside scripts (galleries, sliders)
    for m in re.finditer(
        r"https?://[^\s\"'<>\\]+\.(?:jpe?g|png|webp)(?:\?[^\s\"'<>\\]*)?",
        raw_html,
        re.I,
    ):
        add(m.group(0))

    return found


_rembg_session = None


def make_backdrop():
    """Build the uniform background as an RGB image."""
    w, h = CANVAS_W, CANVAS_H
    ys, xs = np.mgrid[0:h, 0:w].astype("float32")
    if BACKGROUND_STYLE == "gold":
        t = (ys / h)[..., None]                     # light on top, deeper below
        top = np.array([224, 194, 118], dtype="float32")     # #e0c276
        bottom = np.array([163, 125, 49], dtype="float32")   # #a37d31
        arr = top * (1 - t) + bottom * t
        d = np.sqrt(((xs - w / 2) / (w / 2)) ** 2 + ((ys - h / 2) / (h / 2)) ** 2)
        arr *= (1 - 0.18 * np.clip(d, 0, 1.2) ** 2)[..., None]   # soft vignette
    else:                                            # "blackgold"
        d = np.sqrt(((xs - w / 2) / (w / 2)) ** 2 + ((ys - h * 0.55) / (h / 2)) ** 2)
        t = (np.clip(d / 1.15, 0, 1) ** 1.3)[..., None]
        centre = np.array([215, 173, 84], dtype="float32")   # #d7ad54
        edge = np.array([8, 8, 8], dtype="float32")          # #080808
        arr = centre * (1 - t) + edge * t
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), "RGB")


REJECTED = set()
BLOCKLIST = set()
REJECT_COUNTS = {}
LAST_REJECT_REASON = ""


def reject(reason):
    """Remember why a photo was skipped, so the log can show it."""
    global LAST_REJECT_REASON
    LAST_REJECT_REASON = reason
    REJECT_COUNTS[reason] = REJECT_COUNTS.get(reason, 0) + 1
    return None


def load_photo_lists():
    try:
        REJECTED.update(json.loads(PHOTO_CACHE.read_text()).get("rejected", []))
    except (OSError, ValueError):
        pass
    try:
        for line in BLOCKLIST_FILE.read_text().splitlines():
            if not line.strip().startswith("#"):
                BLOCKLIST.update(re.findall(r"\b[0-9a-f]{12}\b", line))
    except OSError:
        pass


def save_rejected():
    PHOTO_CACHE.write_text(json.dumps({"rejected": sorted(REJECTED)}))


_detector = None
_detector_failed = False


def car_gate(img):
    """
    Use an object detector to find the main vehicle and return a mask that
    covers only that vehicle (a little grown). This is what separates the
    main car from other cars parked behind or beside it.
      None  -> detector not available; carry on without it
      False -> no vehicle found in the photo
      mask  -> boolean array, True where the main vehicle is
    """
    global _detector, _detector_failed
    if YOLO is None or _detector_failed:
        return None
    try:
        if _detector is None:
            _detector = YOLO(DETECTOR_MODEL)
        res = _detector.predict(img, conf=0.25, classes=[2, 5, 7], verbose=False)[0]
        if res.masks is None or len(res.masks.xy) == 0:
            return False
        width, height = img.size
        best, best_area = None, 0
        for poly in res.masks.xy:
            if len(poly) < 3:
                continue
            canvas = Image.new("L", (width, height), 0)
            ImageDraw.Draw(canvas).polygon([tuple(pt) for pt in poly.tolist()], fill=255)
            m = np.asarray(canvas) > 0
            area = int(m.sum())
            if area > best_area:
                best, best_area = m, area
        if best is None:
            return False
        if ndimage is not None:
            best = ndimage.binary_dilation(best, iterations=GATE_PX)
        return best
    except Exception as exc:
        print(f"    car detector problem ({exc}); continuing without it")
        _detector_failed = True
        return None


def replace_background(img, relaxed=False):
    """
    Cut the car out of the photo and place it on the uniform backdrop.
    Returns (image, score), or None if the photo is not a clean exterior
    shot of a single car. A higher score means a cleaner shot.
    relaxed=True is the fallback for cars with no clean photo: it accepts
    imperfect shots (but always scores them below MIN_SCORE).
    """
    global _rembg_session
    if _rembg_session is None:
        _rembg_session = new_session("isnet-general-use")

    cutout = rembg_remove(img, session=_rembg_session).convert("RGBA")
    alpha = np.asarray(cutout.split()[3]).copy()
    mask = alpha > MASK_ALPHA

    gate = car_gate(img)
    if gate is False:
        return reject("no car detected in the photo")
    if gate is not None:
        mask = mask & gate
        alpha = np.where(gate, alpha, 0)

    car_px = int(mask.sum())
    share = car_px / mask.size
    if not ((0.06 if relaxed else MIN_CAR_SHARE) <= share <= MAX_CAR_SHARE):
        return reject("car too small or too big in the frame")

    # Erase the dealer-logo corner; skip the photo if the car itself is there.
    if WATERMARK_BOX:
        rows, cols = mask.shape
        bx0, by0, bx1, by1 = WATERMARK_BOX
        box = np.zeros_like(mask)
        box[int(by0 * rows):int(by1 * rows), int(bx0 * cols):int(bx1 * cols)] = True
        if (mask & box).sum() / max(car_px, 1) > MAX_WATERMARK_OVERLAP:
            return reject("car reaches into the logo corner")
        mask = mask & ~box
        alpha = np.where(box, 0, alpha)
        car_px = int(mask.sum())
        if car_px == 0:
            return reject("nothing left after erasing the logo corner")

    # Keep only the main vehicle: shrink the mask so objects touching it
    # (other cars, signs) come apart, keep the biggest piece, grow it back.
    keep = mask
    if ndimage is not None:
        core = ndimage.binary_erosion(mask, iterations=ERODE_PX)
        labels, count = ndimage.label(core)
        if count == 0:
            return reject("no solid car shape found")
        sizes = ndimage.sum(core, labels, range(1, count + 1))
        main = labels == (int(np.argmax(sizes)) + 1)
        keep = ndimage.binary_dilation(main, iterations=ERODE_PX + 2) & mask

    removed = 1 - int(keep.sum()) / max(car_px, 1)
    if removed > MAX_REMOVED_SHARE and not relaxed:
        return reject("other cars or objects in the frame")                     # several vehicles or clutter

    ys, xs = np.where(keep)
    if len(xs) == 0:
        return reject("empty cutout")
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    height, width = keep.shape
    if not relaxed and (x0 <= EDGE_PX or x1 >= width - 1 - EDGE_PX):
        return reject("car cut off by the side of the photo")                     # car is cut off by the side of the photo
    ratio = (x1 - x0 + 1) / (y1 - y0 + 1)
    if not relaxed and not (MIN_RATIO <= ratio <= MAX_RATIO):
        return reject("not a normal car shape (too tall or too wide)")                     # not a normal exterior car shape

    cutout.putalpha(Image.fromarray(np.where(keep, alpha, 0).astype("uint8")))
    cutout = cutout.crop((x0, y0, x1 + 1, y1 + 1))

    scale = min(CANVAS_W * 0.88 / cutout.width, CANVAS_H * 0.74 / cutout.height)
    cutout = cutout.resize(
        (max(1, round(cutout.width * scale)), max(1, round(cutout.height * scale))),
        Image.LANCZOS,
    )

    canvas = make_backdrop().convert("RGBA")
    x = (CANVAS_W - cutout.width) // 2
    bottom = round(CANVAS_H * 0.90)
    y = bottom - cutout.height

    # Soft shadow under the car so it doesn't look pasted on
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    sw, sh = round(cutout.width * 0.92), round(CANVAS_H * 0.07)
    ImageDraw.Draw(shadow).ellipse(
        [(CANVAS_W - sw) // 2, bottom - sh // 2, (CANVAS_W + sw) // 2, bottom + sh // 2],
        fill=(0, 0, 0, 150),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(12))
    canvas = Image.alpha_composite(canvas, shadow)
    canvas.alpha_composite(cutout, (x, y))

    score = int(max(0, min(99, 100 * (1 - removed) - 80 * abs(share - 0.35))))
    if relaxed:
        score = max(1, min(score, MIN_SCORE - 1))
    return canvas.convert("RGB"), score


def save_image(session, url, dest_dir, relaxed=False):
    """
    Download one photo into dest_dir. Returns (file name, score), or None.
    The file name is <hash of source URL and style>-<score>.jpg, so a photo
    already processed on a previous run is not downloaded or processed again.
    """
    key = f"{url}|{BACKGROUND_STYLE}|{STYLE_VERSION}" + ("|relaxed" if relaxed else "")
    stem = hashlib.sha1(key.encode()).hexdigest()[:12]
    if stem in BLOCKLIST or stem in REJECTED:
        return None
    if dest_dir.exists():
        for existing in dest_dir.glob(stem + "-*.jpg"):
            return existing.name, int(existing.stem.split("-")[1])

    try:
        r = session.get(url, headers=HEADERS, timeout=30)
    except requests.RequestException as exc:
        print(f"    photo failed: {exc}")
        return None
    ctype = r.headers.get("Content-Type", "")
    if r.status_code != 200 or not ctype.lower().startswith("image/"):
        print(f"    photo skipped (HTTP {r.status_code}, {ctype or 'no type'})")
        return None
    data = r.content
    if not data or len(data) > MAX_DOWNLOAD_BYTES:
        return None

    score = 50
    if Image is not None:
        try:
            img = Image.open(io.BytesIO(data))
            if img.mode in ("P", "LA", "RGBA"):
                rgba = img.convert("RGBA")
                flat = Image.new("RGB", rgba.size, (255, 255, 255))
                flat.paste(rgba, mask=rgba.split()[3])
                img = flat
            img = img.convert("RGB")
            if img.width < 300:        # tiny images are icons or thumbnails
                REJECTED.add(stem)
                print(f"    photo skipped: only a thumbnail ({img.width}px wide)")
                return None
            if img.width > MAX_WIDTH:
                height = round(img.height * MAX_WIDTH / img.width)
                img = img.resize((MAX_WIDTH, height), Image.LANCZOS)
            if BACKGROUND_STYLE != "none":
                result = replace_background(img, relaxed=relaxed)
                if result is None:
                    print(f"    photo skipped: {LAST_REJECT_REASON}")
                    REJECTED.add(stem)
                    return None
                img, score = result
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=80, optimize=True)
            data = buf.getvalue()
        except Exception as exc:
            print(f"    photo could not be processed: {exc}")
            return None
    elif "jpeg" not in ctype.lower():
        return None                    # without Pillow, keep only real JPEGs

    dest_dir.mkdir(parents=True, exist_ok=True)
    name = f"{stem}-{score:02d}.jpg"
    (dest_dir / name).write_bytes(data)
    return name, score


def save_photos(session, car_id, source_urls):
    """Process several photos, keep the cleanest ones, return public links."""
    car_dir = PHOTO_DIR / car_id
    candidates = []
    for url in source_urls[:MAX_CANDIDATES]:
        time.sleep(DELAY_SECONDS / 2)
        result = save_image(session, url, car_dir)
        if result and result[1] >= MIN_SCORE and result[0] not in [c[0] for c in candidates]:
            candidates.append(result)

    candidates.sort(key=lambda c: -c[1])           # cleanest photo first
    keep = [name for name, _ in candidates[:MAX_PHOTOS]]

    # Fallback: no clean photo passed, so take the best imperfect ones
    # rather than leaving the car with no picture at all.
    if not keep:
        relaxed_found = []
        for url in source_urls[:MAX_CANDIDATES]:
            time.sleep(DELAY_SECONDS / 2)
            result = save_image(session, url, car_dir, relaxed=True)
            if result and result[0] not in [c[0] for c in relaxed_found]:
                relaxed_found.append(result)
        relaxed_found.sort(key=lambda c: -c[1])
        keep = [name for name, _ in relaxed_found[:2]]
        if keep:
            print(f"    no clean photo - using {len(keep)} fallback photo(s)")

    if car_dir.exists():
        for old in car_dir.iterdir():
            if old.name not in keep:
                old.unlink()

    return [f"{PUBLIC_PHOTO_BASE}/{car_id}/{name}" for name in keep]


def make_preview(vehicles):
    """One picture with the first photo of every car, to review at a glance."""
    if Image is None or not vehicles:
        return
    cols, tw, th = 5, 360, 203
    rows = -(-len(vehicles) // cols)
    sheet = Image.new("RGB", (cols * tw, rows * th), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for i, v in enumerate(vehicles):
        x, y = (i % cols) * tw, (i // cols) * th
        path = None
        if v["photos"] and "/photos/" in v["photos"][0]:
            path = PHOTO_DIR / v["photos"][0].split("/photos/", 1)[1]
        if path is not None and path.exists():
            sheet.paste(Image.open(path).convert("RGB").resize((tw, th)), (x, y))
        else:
            draw.text((x + 12, y + th // 2), "NO PHOTO", fill=(215, 173, 84))
        label = v["title"][:40]
        draw.text((x + 7, y + 7), label, fill=(0, 0, 0))
        draw.text((x + 6, y + 6), label, fill=(255, 255, 255))
    sheet.save("preview.jpg", "JPEG", quality=80)


def remove_sold_car_photos(active_ids):
    if not PHOTO_DIR.exists():
        return
    for folder in PHOTO_DIR.iterdir():
        if folder.is_dir() and folder.name not in active_ids:
            shutil.rmtree(folder)
            print(f"Removed photos for sold car {folder.name}")


def dump_debug(source_id, html):
    """Save a car page and list every picture-like or data-feed link in it."""
    Path("debug").mkdir(exist_ok=True)
    Path(f"debug/car-{source_id}.html").write_text(html, encoding="utf-8")
    pics = sorted(set(re.findall(
        r"""["'(]([^"'()\s]*(?:picture|photo|image|gallery|slide)[^"'()\s]*)""",
        html, re.I)))
    feeds = sorted(set(re.findall(
        r"""["'](/[^"'\s]*(?:api|json|ajax|photos|images)[^"'\s]*)["']""",
        html, re.I)))
    print(f"    DEBUG car page {source_id}: {len(html)} characters")
    print(f"    DEBUG {len(pics)} picture-like strings:")
    for item in pics[:40]:
        print(f"      {item}")
    print(f"    DEBUG {len(feeds)} possible data links:")
    for item in feeds[:20]:
        print(f"      {item}")


def main():
    session = requests.Session()

    print(f"Fetching list: {LIST_URL}")
    list_html = fetch(session, LIST_URL)
    if not list_html:
        print("ERROR: could not load the list page. Feed left unchanged.")
        sys.exit(1)

    cars = list_cars(list_html)
    print(f"Found {len(cars)} car links on the list page.")
    if Image is None:
        print("NOTE: Pillow is not installed, so photos will not be resized.")
    print("Script version: 7 (relaxed fallback + debug dump)")
    print("Car detector:", "ON" if YOLO is not None else "OFF (not installed)")
    load_photo_lists()
    if BACKGROUND_STYLE != "none" and (
        Image is None or np is None or rembg_remove is None
    ):
        print(
            "ERROR: background replacement needs pillow, numpy and rembg "
            "installed. Feed left unchanged."
        )
        sys.exit(1)

    vehicles = []
    for index, (source_id, url) in enumerate(cars.items(), start=1):
        time.sleep(DELAY_SECONDS)
        html = fetch(session, url)
        if not html:
            print(f"[{index}/{len(cars)}] SKIPPED {url} (could not load)")
            continue

        if index <= 3:
            dump_debug(source_id, html)

        soup = BeautifulSoup(html, "html.parser")
        title = find_title(soup)
        if not title:
            print(f"[{index}/{len(cars)}] SKIPPED {url} (no title found)")
            continue

        car_id = f"ae-{source_id}"
        year, make, model = split_title(title)
        source_photos = find_photos(soup, url, html)
        photos = save_photos(session, car_id, source_photos)
        mileage = find_mileage(soup.get_text(" ", strip=True))

        vehicles.append(
            {
                "id": car_id,
                "title": title,
                "year": year,
                "make": make,
                "model": model,
                "mileage": mileage,
                "photos": photos,
                "pricing_label": "CALL FOR PRICING",
            }
        )
        print(
            f"[{index}/{len(cars)}] {title} | found {len(source_photos)}, "
            f"saved {len(photos)} photos | miles: {mileage}"
        )

    save_rejected()
    if REJECT_COUNTS:
        print("\nWhy photos were skipped this run:")
        for reason, count in sorted(REJECT_COUNTS.items(), key=lambda kv: -kv[1]):
            print(f"  {count:4d}  {reason}")
    with_photos = sum(1 for v in vehicles if v["photos"])
    print(f"\nBuilt {len(vehicles)} vehicles, {with_photos} with photos.")

    if len(vehicles) < MIN_EXPECTED:
        print(
            f"ERROR: only {len(vehicles)} cars (expected at least {MIN_EXPECTED}). "
            "Feed left unchanged."
        )
        sys.exit(1)
    if with_photos == 0:
        print("WARNING: no photos were saved for any car.")

    remove_sold_car_photos({v["id"] for v in vehicles})
    vehicles.sort(key=lambda v: (v["make"].lower(), v["model"].lower(), v["year"] or 0))

    make_preview(vehicles)

    # Final safety check: nothing in the feed may mention the source site.
    feed_text = json.dumps(vehicles).lower()
    if "sunshineauto" in feed_text or "sunshine auto" in feed_text:
        print("ERROR: source dealer name found in the feed. Feed left unchanged.")
        sys.exit(1)

    # Don't rewrite the file (and create a pointless commit) if nothing changed.
    try:
        with open(OUTPUT, encoding="utf-8") as f:
            if json.load(f).get("vehicles") == vehicles:
                print("No changes since last run.")
                return
    except (OSError, ValueError):
        pass

    data = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(vehicles),
        "vehicles": vehicles,
    }
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
