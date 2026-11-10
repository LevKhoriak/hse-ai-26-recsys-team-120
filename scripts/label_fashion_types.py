"""Типы товаров Fashion: справочник Amazon и разметка названий LLM.

У Amazon_Fashion пустые `categories`, но в Clothing_Shoes_and_Jewelry те же
по смыслу товары (main_category = AMAZON FASHION) лежат с путём каталога.
Пересечения по parent_asin почти нет, поэтому Clothing служит проверочной
выборкой: путь каталога -> тип даёт эталон, а LLM размечает названия.

Запуск из корня репозитория:

    # 1. срезы файла Clothing (5 x 100 МБ) и эталон по пути каталога
    python scripts/label_fashion_types.py reference
    # 2. разметка CSV с колонками parent_asin, title
    python scripts/label_fashion_types.py label IN.csv OUT.csv
    # 3. точность разметки эталона -> reports/fashion_types_reference.csv
    python scripts/label_fashion_types.py label \\
        data/cache/fashion_types/reference_in.csv \\
        data/cache/fashion_types/reference_llm.csv
    python scripts/label_fashion_types.py evaluate

Модель: Claude Haiku 5.5 через агрегатор Polza (OpenAI-совместимый API), effort
low, 50 названий на запрос, ответ строго по JSON-схеме со списком типов. Ключ
берётся из `POLZA_API_KEY` (окружение или `.env`). Результаты LLM не
воспроизводятся побайтно, поэтому метки сохранены в `reports/`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import urllib3

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/fashion_types"

CLOTHING_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
    "raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl"
)
CLOTHING_BYTES = 17_960_532_903
SLICE_BYTES = 100_000_000
SLICES = (0, 20, 40, 60, 80)  # проценты от начала файла

API_URL = "https://polza.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-haiku-5.5"
EFFORT = "low"
BATCH = 50
WORKERS = 8

# тип -> что в него входит; описание идёт в промпт
TYPES = {
    "tops": "T-shirts, tees, blouses, shirts (incl. dress shirts), tank tops, "
    "camisoles worn as tops, tunics, polos, henleys, jerseys, crop tops",
    "sweaters_hoodies": "sweaters, cardigans, pullovers, hoodies, sweatshirts, "
    "fleece tops, shrugs",
    "outerwear": "coats, jackets, parkas, vests and gilets, windbreakers, raincoats",
    "suits_blazers": "suits, blazers, sport coats",
    "dresses": "dresses and gowns (not dress shirts, dress shoes or dress socks)",
    "skirts": "skirts and skorts",
    "pants": "pants, trousers, jeans, leggings, joggers, sweatpants",
    "shorts": "shorts (not swim trunks or board shorts)",
    "one_pieces": "jumpsuits, rompers, overalls, bodysuits, baby onesies",
    "clothing_sets": "sets or outfits of different garments (e.g. top + pants), "
    "tracksuits; not pajama sets",
    "swimwear": "swimsuits, bikinis, swim trunks, board shorts, rash guards, "
    "swim cover-ups",
    "underwear": "bras, panties, briefs, boxers, thongs, lingerie, shapewear, "
    "undershirts",
    "sleepwear": "pajamas, pajama sets, nightgowns, sleepshirts, robes",
    "socks_hosiery": "socks, tights, stockings, hosiery, leg warmers, slipper socks",
    "costumes": "costumes, cosplay outfits, costume masks and costume accessories",
    "shoes": "all footwear: shoes, sneakers, boots, sandals, slippers, heels, "
    "flats, loafers, clogs",
    "jewelry": "necklaces, earrings, bracelets, rings, pendants, anklets, brooches, "
    "body jewelry, charms, jewelry sets",
    "watches": "wrist watches and smartwatches",
    "watch_bands": "watch bands, watch straps and other watch accessories "
    "(not the watch itself)",
    "eyewear": "sunglasses, eyeglasses, reading glasses, eyewear frames",
    "hats": "hats, caps, beanies, berets, visors, other headwear",
    "bags_wallets": "handbags, purses, backpacks, totes, crossbody and shoulder bags, "
    "wallets, card cases, luggage, waist packs",
    "scarves_gloves": "scarves, shawls, wraps, gloves, mittens, balaclavas, "
    "neck gaiters, earmuffs",
    "belts": "belts and suspenders",
    "face_masks": "protective or fashion face masks (not costume masks)",
    "other_accessories": "ties, bow ties, cuff links, keychains, hair accessories, "
    "shoelaces, insoles, glasses cases and chains, umbrellas, other small accessories",
    "other": "not a fashion item, or fits none of the types above",
}

# Путь каталога -> тип. Сначала правила на весь путь, потом узлы от самого
# глубокого к корню: первый узел, совпавший с правилом, задаёт тип.
PATH_OVERRIDES = [
    # узлы целиком: "Shoe, Jewelry & Watch Accessories" и "Sunglasses & Eyewear
    # Accessories" это группы, а не сами аксессуары
    (r"\bwatch bands?\b|(^|> )watch accessories\b", "watch_bands"),
    (r"\bcostumes?\b", "costumes"),
    (r"\bswim|\bbikinis?\b|\brash guard", "swimwear"),
    (r"\bpanties\b", "underwear"),  # Panties > Boy Shorts
    (r"\bjewelry sets\b", "jewelry"),
    (r"\btie sets\b", "other_accessories"),
    (r"\bshoe, jewelry & watch accessories\b", "other_accessories"),
    (r"\bkeyrings?\b|\bkeychains?\b|\bshoelaces\b|\binsoles\b", "other_accessories"),
    (
        r"(^|> )eyewear accessories\b|\breplacement sunglass lenses\b"
        r"|\beyeglass cases\b|\beyewear retainers\b",
        "other_accessories",
    ),
]
NODE_RULES = [
    (r"\bsleep|\brobes?\b|\bnightgown|\bpajama", "sleepwear"),
    (
        r"\bbras?\b|\bpanties\b|\bunderwear\b|\bbriefs\b|\bboxers?\b|\bthongs?\b"
        r"|\bg-strings?\b|\blingerie\b|\bundershirts?\b|\bshapewear\b",
        "underwear",
    ),
    (
        r"\bpants\b|\bsweatpants\b|\bjeans\b|\btrousers\b|\bjoggers\b|\bleggings\b"
        r"|\bscrub bottoms\b|\bcargo\b",
        "pants",
    ),
    (
        r"\bsocks?\b|\bhosiery\b|\btights\b|\bleg warmers\b|\bstockings\b",
        "socks_hosiery",
    ),
    (r"\bmasks?\b", "face_masks"),
    (
        r"\bbalaclavas?\b|\bneck gaiters?\b|\bscarves\b|\bscarf\b|\bwraps\b"
        r"|\bpashminas?\b|\bgloves\b|\bmittens\b|\bearmuffs\b",
        "scarves_gloves",
    ),
    (r"\btracksuits?\b|\bsets\b|\boutfits?\b", "clothing_sets"),
    (
        r"\bjumpsuits?\b|\brompers?\b|\boveralls\b|\bbodysuits?\b|\bone-pieces\b",
        "one_pieces",
    ),
    (
        r"\bt-shirts?\b|\btees?\b|\bshirts?\b|\bblouses?\b|\btops?\b|\btanks?\b"
        r"|\bcamis\b|\bcamisoles\b|\btunics?\b|\bpolos?\b|\bhenleys?\b|\bjerseys?\b",
        "tops",
    ),
    (
        r"\bhoodies\b|\bsweatshirts?\b|\bsweaters?\b|\bcardigans?\b|\bpullovers?\b"
        r"|\bfleece\b|\bshrugs?\b",
        "sweaters_hoodies",
    ),
    (r"\bblazers?\b|\bsuits?\b|\bsport coats?\b|\bsuiting\b", "suits_blazers"),
    (
        r"\bcoats?\b|\bjackets?\b|\bvests?\b|\bparkas?\b|\bwindbreakers?\b"
        r"|\braincoats?\b|\bdown\b|\bshearling\b|\bfur\b|\bouterwear\b",
        "outerwear",
    ),
    # только "Dresses": узел "Dress" бывает под Pants и Shoes (dress pants/shoes)
    (r"\bdresses\b", "dresses"),
    (r"\bskirts?\b|\bskorts?\b", "skirts"),
    (r"\bshorts\b", "shorts"),
    (
        r"\bshoes?\b|\bsneakers?\b|\bboots?\b|\bsandals?\b|\bslippers?\b|\bheels?\b"
        r"|\bpumps?\b|\bflats\b|\bloafers?\b|\boxfords?\b|\bclogs?\b|\bmules?\b"
        r"|\bflip-flops?\b|\bslides\b|\bwedges\b|\bplatforms\b|\bmoccasins?\b"
        r"|\bespadrilles?\b|\bcleats?\b|\bfootwear\b",
        "shoes",
    ),
    (
        r"\bjewelry\b|\bearrings?\b|\bnecklaces?\b|\bbracelets?\b|\brings?\b"
        r"|\bpendants?\b|\banklets?\b|\bbrooches\b|\bpins\b|\bcharms?\b|\bchokers?\b"
        r"|\bchains\b|\bbangles?\b",
        "jewelry",
    ),
    (r"\bwatch(es)?\b", "watches"),
    (r"\bsunglasses\b|\beyeglasses\b|\beyewear\b|\bglasses\b", "eyewear"),
    (
        r"\bhats?\b|\bcaps?\b|\bbeanies\b|\bskullies\b|\bfedoras\b|\bberets?\b"
        r"|\bvisors?\b|\bheadwear\b",
        "hats",
    ),
    (
        r"\bbags?\b|\bbackpacks?\b|\bdaypacks?\b|\bwallets?\b|\bpurses?\b"
        r"|\bhandbags?\b|\btotes?\b|\bclutch(es)?\b|\bsatchels?\b|\bhobo\b"
        r"|\bwristlets?\b|\bluggage\b|\bcard cases\b|\bwaist packs?\b",
        "bags_wallets",
    ),
    (r"\bbelts?\b|\bsuspenders\b", "belts"),
    (
        r"\bties\b|\bneckties\b|\bcummerbunds\b|\bpocket squares\b|\bcuff links\b"
        r"|\bumbrellas\b|\bhair accessories\b",
        "other_accessories",
    ),
]


def path_type(path: list[str]) -> str | None:
    """Тип по пути каталога Amazon или None, если путь ничего не говорит."""
    nodes = [n.lower() for n in path[1:]]  # корень "Clothing, Shoes & Jewelry"
    whole = " > ".join(nodes)
    for pattern, name in PATH_OVERRIDES:
        if re.search(pattern, whole):
            return name
    for node in reversed(nodes):
        for pattern, name in NODE_RULES:
            if re.search(pattern, node):
                return name
    return None


# --- справочник из Clothing ------------------------------------------------------


def download_slices(folder: Path) -> list[Path]:
    """Пять кусков по 100 МБ из разных мест файла; весь файл 18 ГБ."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for pct in SLICES:
        path = folder / f"slice_{pct:02d}.jsonl"
        if not path.exists() or path.stat().st_size < SLICE_BYTES:
            start = CLOTHING_BYTES * pct // 100
            byte_range = f"{start}-{start + SLICE_BYTES}"
            cmd = ["curl", "-sfL", "-r", byte_range, "-o", str(path), CLOTHING_URL]
            subprocess.run(cmd, check=True)
        paths.append(path)
    return paths


def reference(per_type: int, seed: int) -> Path:
    rows: dict[str, list[dict[str, str]]] = {}
    for path in download_slices(CACHE / "clothing_slices"):
        # первая и последняя строки куска обрезаны
        for line in path.read_bytes().split(b"\n")[1:-1]:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            title = " ".join((rec.get("title") or "").split())
            categories = rec.get("categories") or []
            if rec.get("main_category") != "AMAZON FASHION" or not title:
                continue
            name = path_type(categories)
            if name is None:
                continue
            rows.setdefault(name, []).append(
                {
                    "parent_asin": rec["parent_asin"],
                    "title": title,
                    "path": " > ".join(categories[1:]),
                    "true_type": name,
                }
            )
    rng = random.Random(seed)
    out = CACHE / "reference.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, ["parent_asin", "title", "path", "true_type"])
        writer.writeheader()
        for name in sorted(rows):
            items = rows[name]
            rng.shuffle(items)
            writer.writerows(items[:per_type])
            print(f"{name}: {len(items):,} в срезах, взяли {min(per_type, len(items))}")
    return out


# --- разметка LLM ----------------------------------------------------------------

SYSTEM = (
    "You classify Amazon fashion product titles into exactly one item type.\n\n"
    "Types:\n"
    + "\n".join(f"- {name}: {desc}" for name, desc in TYPES.items())
    + "\n\nRules:\n"
    "- Label the main product being sold, not what it is used with: a watch band is "
    "watch_bands, a glasses case is other_accessories, dress shoes are shoes.\n"
    "- Several different garments sold together are clothing_sets; a multipack of "
    "the same item takes that item's type.\n"
    "- Use other only when no type fits.\n"
    "Return one label for every input item, keyed by its id."
)
SCHEMA = {
    "name": "item_types",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "type": {"type": "string", "enum": list(TYPES)},
                    },
                    "required": ["id", "type"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["labels"],
        "additionalProperties": False,
    },
}


def api_key() -> str:
    key = os.environ.get("POLZA_API_KEY")
    if not key and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("POLZA_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        raise SystemExit("нет POLZA_API_KEY в окружении или .env")
    return key


HTTP = urllib3.PoolManager(retries=False, timeout=urllib3.Timeout(30, read=180))


def call(titles: list[str], key: str, retries: int = 5) -> tuple[list[str], dict]:
    """Метки для пачки названий в исходном порядке и usage ответа."""
    body = {
        "model": MODEL,
        "max_tokens": 4000,
        "reasoning": {"effort": EFFORT},
        "response_format": {"type": "json_schema", "json_schema": SCHEMA},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    [{"id": i, "title": t[:300]} for i, t in enumerate(titles)],
                    ensure_ascii=False,
                ),
            },
        ],
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for attempt in range(retries):
        resp = HTTP.request("POST", API_URL, body=json.dumps(body), headers=headers)
        if resp.status in (408, 429) or resp.status >= 500:
            time.sleep(2**attempt + random.random())
            continue
        data = resp.json()
        if resp.status != 200:
            raise RuntimeError(f"{resp.status}: {data}")
        content = data["choices"][0]["message"]["content"]
        try:
            labels = {x["id"]: x["type"] for x in json.loads(content)["labels"]}
        except (ValueError, KeyError, TypeError):
            time.sleep(1)
            continue
        if set(labels) != set(range(len(titles))):
            continue  # пропуск или лишний id: просим пачку заново
        return [labels[i] for i in range(len(titles))], data.get("usage", {})
    raise RuntimeError("пачка не размечена после повторов")


def label(src: Path, dst: Path) -> None:
    key = api_key()
    with src.open() as f:
        items = [(r["parent_asin"], r["title"]) for r in csv.DictReader(f)]
    done_path = CACHE / f"llm_{dst.stem}.jsonl"
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, str] = {}
    if done_path.exists():
        for line in done_path.read_text().splitlines():
            rec = json.loads(line)
            done[rec["parent_asin"]] = rec["llm_type"]
    todo = [(a, t) for a, t in items if a not in done]
    batches = [todo[i : i + BATCH] for i in range(0, len(todo), BATCH)]
    print(f"{len(items):,} товаров, уже размечено {len(done):,}, пачек {len(batches)}")
    cost = 0.0

    def run(batch: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], dict]:
        labels, usage = call([t for _, t in batch], key)
        return [(a, lab) for (a, _), lab in zip(batch, labels, strict=True)], usage

    with done_path.open("a") as log, ThreadPoolExecutor(WORKERS) as pool:
        for n, (pairs, usage) in enumerate(pool.map(run, batches), start=1):
            for asin, lab in pairs:
                done[asin] = lab
                log.write(json.dumps({"parent_asin": asin, "llm_type": lab}) + "\n")
            log.flush()
            cost += float(usage.get("cost_rub") or 0)
            if n % 20 == 0 or n == len(batches):
                print(f"  {n}/{len(batches)} пачек, {cost:.2f} руб.")
    with dst.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parent_asin", "llm_type"])
        writer.writerows((a, done[a]) for a, _ in items)
    print(f"готово: {dst}; стоимость этого запуска {cost:.2f} руб.")


def evaluate(ref_path: Path, llm_path: Path, out: Path) -> None:
    """Сравнить разметку LLM с типом по пути каталога; сохранить объединение."""
    with llm_path.open() as f:
        llm = {r["parent_asin"]: r["llm_type"] for r in csv.DictReader(f)}
    with ref_path.open() as f:
        rows = [{**r, "llm_type": llm[r["parent_asin"]]} for r in csv.DictReader(f)]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ok = sum(r["true_type"] == r["llm_type"] for r in rows)
    print(f"{len(rows)} товаров, совпало с путём каталога {ok / len(rows):.1%}")
    true_n = Counter(r["true_type"] for r in rows)
    pred_n = Counter(r["llm_type"] for r in rows)
    hits = Counter(r["true_type"] for r in rows if r["true_type"] == r["llm_type"])
    print(f"{'тип':18} recall precision")
    for name in sorted(true_n, key=lambda t: hits[t] / true_n[t]):
        prec = hits[name] / pred_n[name] if pred_n[name] else float("nan")
        print(f"{name:18} {hits[name] / true_n[name]:6.2f} {prec:9.2f}")
    errors = Counter(
        (r["true_type"], r["llm_type"]) for r in rows if r["true_type"] != r["llm_type"]
    )
    print("частые ошибки (каталог -> LLM):")
    for (t, p), n in errors.most_common(10):
        print(f"  {t} -> {p}: {n}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    ref = sub.add_parser("reference", help="эталон из среза Clothing")
    ref.add_argument("--per-type", type=int, default=100)
    ref.add_argument("--seed", type=int, default=0)
    lab = sub.add_parser("label", help="разметить CSV parent_asin,title")
    lab.add_argument("src", type=Path)
    lab.add_argument("dst", type=Path)
    ev = sub.add_parser("evaluate", help="сравнить разметку эталона с путём каталога")
    ev.add_argument("--reference", type=Path, default=CACHE / "reference.csv")
    ev.add_argument("--llm", type=Path, default=CACHE / "reference_llm.csv")
    ev.add_argument(
        "--out", type=Path, default=ROOT / "reports/fashion_types_reference.csv"
    )
    args = parser.parse_args()
    if args.cmd == "reference":
        print(reference(args.per_type, args.seed))
    elif args.cmd == "label":
        label(args.src, args.dst)
    else:
        evaluate(args.reference, args.llm, args.out)


if __name__ == "__main__":
    main()
