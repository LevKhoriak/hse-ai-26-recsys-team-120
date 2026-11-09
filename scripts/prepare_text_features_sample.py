"""Фиксированная выборка Grocery без S3. Запускать из корня репозитория."""

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
REVISION = "2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e"
SOURCE_SHA256 = "ed9b24ea707f7010b5cb321462a8e896ad1fcd59620fbc55f05084c3da59842e"
SOURCE_SIZE = 1_382_105_675
CATEGORY = "Grocery_and_Gourmet_Food"
URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/"
    f"{REVISION}/raw/meta_categories/meta_{CATEGORY}.jsonl"
)


def clean_text(value):
    return " ".join(value.split()) if isinstance(value, str) else ""


def category_label(value):
    # Второй уровень пути: например, Beverages. Метку не подаём энкодеру.
    if isinstance(value, list) and len(value) > 1:
        return clean_text(value[1])
    return ""


def prepare(source, size=2000, seed=42):
    rng = random.Random(seed)
    counts = Counter(
        dict.fromkeys(
            [
                "source_rows",
                "invalid_json",
                "invalid_record",
                "features_non_list",
                "missing_id",
                "missing_title",
                "missing_features",
                "duplicate_eligible_id",
                "eligible_unique",
            ],
            0,
        )
    )
    labels = Counter()
    seen = set()
    sample = []
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for line in stream:
            digest.update(line)
            counts["source_rows"] += 1
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                counts["invalid_json"] += 1
                continue
            if not isinstance(row, dict):
                counts["invalid_record"] += 1
                continue
            asin = clean_text(row.get("parent_asin"))
            title = clean_text(row.get("title"))
            raw_features = row.get("features")
            if not isinstance(raw_features, list):
                counts["features_non_list"] += 1
            features = (
                [clean_text(x) for x in raw_features]
                if isinstance(raw_features, list)
                else []
            )
            features = [x for x in features if x]
            if not asin:
                counts["missing_id"] += 1
                continue
            if not title:
                counts["missing_title"] += 1
                continue
            if not features:
                counts["missing_features"] += 1
                continue
            if asin in seen:
                counts["duplicate_eligible_id"] += 1
                continue
            # Первая пригодная карточка для каждого parent_asin.
            seen.add(asin)
            counts["eligible_unique"] += 1
            label = category_label(row.get("categories"))
            labels[label] += 1
            selected = {
                "parent_asin": asin,
                "title": title,
                "features": features,
                "categories": row.get("categories")
                if isinstance(row.get("categories"), list)
                else [],
                "label": label,
                "store": clean_text(row.get("store")),
                "title_raw": row.get("title"),
                "features_raw_json": json.dumps(raw_features, ensure_ascii=False),
            }
            index = counts["eligible_unique"]
            if len(sample) < size:
                sample.append(selected)
            else:
                position = rng.randrange(index)
                if position < size:
                    sample[position] = selected
    if digest.hexdigest() != SOURCE_SHA256 or source.stat().st_size != SOURCE_SIZE:
        raise ValueError("Источник не совпадает с зафиксированным SHA256/размером")
    rejected = [
        "invalid_json",
        "invalid_record",
        "missing_id",
        "missing_title",
        "missing_features",
        "duplicate_eligible_id",
    ]
    assert counts["source_rows"] == counts["eligible_unique"] + sum(
        counts[x] for x in rejected
    )
    frame = pd.DataFrame(sample).sort_values("parent_asin").reset_index(drop=True)
    assert frame.parent_asin.is_unique and len(frame) == min(
        size, counts["eligible_unique"]
    )
    return frame, counts, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / f"data/raw/text_features/meta_{CATEGORY}.jsonl",
    )
    parser.add_argument("--size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.size < 2:
        parser.error("--size должен быть не меньше 2")
    source = args.source
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        temporary = source.with_suffix(source.suffix + ".part")
        if not temporary.exists() or temporary.stat().st_size != SOURCE_SIZE:
            with requests.get(URL, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with temporary.open("wb") as out:
                    for chunk in response.iter_content(1024 * 1024):
                        out.write(chunk)
        # prepare проверит хеш до записи результатов.
        frame, counts, labels = prepare(temporary, args.size, args.seed)
        temporary.replace(source)
    else:
        frame, counts, labels = prepare(source, args.size, args.seed)

    cache = ROOT / "data/cache/text_features"
    reports = ROOT / "reports"
    cache.mkdir(parents=True, exist_ok=True)
    reports.mkdir(exist_ok=True)
    frame.to_parquet(cache / "sample.parquet", index=False)
    frame[["parent_asin"]].assign(category=CATEGORY).to_csv(
        reports / "eda_text_features_sample_ids.csv", index=False
    )
    rows = [
        {
            "section": "source",
            "metric": key,
            "group": "all",
            "value": value,
            "n": counts["source_rows"],
        }
        for key, value in sorted(counts.items())
    ]
    sample_labels = Counter(frame.label)
    for label in sorted(labels):
        rows.extend(
            [
                {
                    "section": "label_coverage",
                    "metric": "eligible_count",
                    "group": label or "missing",
                    "value": labels[label],
                    "n": counts["eligible_unique"],
                },
                {
                    "section": "label_coverage",
                    "metric": "sample_count",
                    "group": label or "missing",
                    "value": sample_labels[label],
                    "n": len(frame),
                },
            ]
        )
    source_audit = pd.DataFrame(rows).assign(category=CATEGORY)
    source_audit.to_csv(cache / "source_audit.csv", index=False)
    source_audit.to_csv(reports / "eda_text_features_audit.csv", index=False)
    config = {
        "category": CATEGORY,
        "source_url": URL,
        "source_revision": REVISION,
        "source_sha256": SOURCE_SHA256,
        "source_bytes": SOURCE_SIZE,
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "sample_size": len(frame),
        "seed": args.seed,
        "selection": (
            "reservoir over unique items with nonempty title and list features; "
            "first eligible record per ID"
        ),
        "label": "categories[1]; missing labels excluded from purity denominators",
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(frame.parent_asin).encode()
        ).hexdigest(),
        "sample_parquet_sha256": hashlib.sha256(
            (cache / "sample.parquet").read_bytes()
        ).hexdigest(),
        "preparation_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "team_s3_comparison": "not performed",
    }
    pd.DataFrame(config.items(), columns=["parameter", "value"]).to_csv(
        cache / "source_config.csv", index=False
    )
    pd.DataFrame(config.items(), columns=["parameter", "value"]).to_csv(
        reports / "eda_text_features_config.csv", index=False
    )
    print(json.dumps(dict(counts), indent=2))
    print(f"Сохранено {len(frame)} товаров; SHA256 источника проверен.")


if __name__ == "__main__":
    main()
