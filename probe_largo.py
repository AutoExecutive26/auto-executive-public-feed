# Looks at largoautomall.com and prints what the site is built like.
# It changes nothing. Run it from GitHub Actions and send me the log.
import re, requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE = "https://largoautomall.com"
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
PATHS = ["", "/inventory", "/inventory/", "/used-inventory", "/vehicles", "/cars",
         "/legacy", "/search", "/used-cars", "/inventory?page=2", "/robots.txt", "/sitemap.xml"]

def get(u):
    try:
        return requests.get(u, headers=H, timeout=30)
    except Exception as e:
        print("   ERROR", e); return None

for p in PATHS:
    url = BASE + p
    r = get(url)
    if r is None: continue
    print(f"\n=== {url} -> {r.status_code}, {len(r.text)} chars, type {r.headers.get('content-type','')[:30]}")
    if r.status_code != 200: continue
    t = r.text
    if p in ("/robots.txt",):
        print(t[:800]); continue
    if p == "/sitemap.xml":
        print(re.findall(r"<loc>(.*?)</loc>", t)[:15]); continue
    soup = BeautifulSoup(t, "html.parser")
    print("   title:", soup.title.string.strip()[:100] if soup.title and soup.title.string else None)
    gen = soup.find("meta", {"name": "generator"})
    print("   generator:", gen.get("content") if gen else None)
    hints = [k for k in ["dealer.com","dealeron","dealercenter","dealersocket","dealerfire","autorevo","carsforsale","dealerinspire","wordpress","wix","squarespace","shopify","vauto","nextjs","__NEXT_DATA__","ng-app","react"] if k.lower() in t.lower()]
    print("   platform hints:", hints)
    links = sorted({urljoin(url, a["href"]) for a in soup.find_all("a", href=True)})
    veh = [l for l in links if re.search(r"/(vehicle|vehicles|inventory|used|detail|vdp|view|car)s?[/-]", l, re.I)]
    print(f"   links: {len(links)} total, {len(veh)} look like vehicle links")
    for l in veh[:12]: print("     ", l)
    imgs = re.findall(r'https?://[^\s"\'<>)]+\.(?:jpg|jpeg|png|webp)', t, re.I)
    print(f"   image urls in html: {len(imgs)}")
    for i in imgs[:5]: print("     ", i)
    print("   JSON-LD/Vehicle mentions:", len(re.findall(r'"@type"\s*:\s*"(?:Vehicle|Car)"', t)))
    print("   script srcs:", [s["src"][:80] for s in soup.find_all("script", src=True)][:6])
