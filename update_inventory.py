#!/usr/bin/env python3
"""
Builds auto_executive_test_inventory.json from the source dealer's plain
"legacy" inventory pages.

1. Reads the full list page and collects every car link.
2. Opens each car page for the title, mileage and photo URLs.
3. Writes everything to the JSON file your Squarespace page reads.

Safety: if far fewer cars than expected are found (for example the source
site blocks us), the existing JSON file is left untouched and the run fails.
"""

import json
import re
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://sunshineautobrokers.com"
LIST_URL = BASE + "/legacy"
OUTPUT = "auto_executive_test_inventory.json"
MIN_EXPECTED = 30          # refuse to overwrite the feed with fewer cars than this
MAX_PHOTOS = 25            # photos kept per car
DELAY_SECONDS = 0.6        # pause between requests, to be polite to the source

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


def clean_title(title):
    words = []
    for w in title.split():
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

    return found[:MAX_PHOTOS]


def main():
    session = requests.Session()

    print(f"Fetching list: {LIST_URL}")
    list_html = fetch(session, LIST_URL)
    if not list_html:
        print("ERROR: could not load the list page. Feed left unchanged.")
        sys.exit(1)

    cars = list_cars(list_html)
    print(f"Found {len(cars)} car links on the list page.")

    vehicles = []
    for index, (source_id, url) in enumerate(cars.items(), start=1):
        time.sleep(DELAY_SECONDS)
        html = fetch(session, url)
        if not html:
            print(f"[{index}/{len(cars)}] SKIPPED {url} (could not load)")
            continue

        soup = BeautifulSoup(html, "html.parser")
        title = find_title(soup)
        if not title:
            print(f"[{index}/{len(cars)}] SKIPPED {url} (no title found)")
            continue

        year, make, model = split_title(title)
        photos = find_photos(soup, url, html)
        mileage = find_mileage(soup.get_text(" ", strip=True))

        vehicles.append(
            {
                "id": f"ae-{source_id}",
                "title": title,
                "year": year,
                "make": make,
                "model": model,
                "mileage": mileage,
                "photos": photos,
                "pricing_label": "CALL FOR PRICING",
            }
        )
        print(f"[{index}/{len(cars)}] {title} | photos: {len(photos)} | miles: {mileage}")
        if index <= 3 and photos:
            print(f"    first photo: {photos[0]}")

    with_photos = sum(1 for v in vehicles if v["photos"])
    print(f"\nBuilt {len(vehicles)} vehicles, {with_photos} with photos.")

    if len(vehicles) < MIN_EXPECTED:
        print(
            f"ERROR: only {len(vehicles)} cars (expected at least {MIN_EXPECTED}). "
            "Feed left unchanged."
        )
        sys.exit(1)
    if with_photos == 0:
        print("WARNING: no photos were found on any car page.")

    vehicles.sort(key=lambda v: (v["make"].lower(), v["model"].lower(), v["year"] or 0))

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
