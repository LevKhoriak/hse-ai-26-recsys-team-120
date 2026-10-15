# Данные

Обработанные данные Amazon Reviews 2023 лежат в S3 (Selectel) в формате Parquet. Этот README — про то, как их читать.

## Что где лежит

```
s3://hse-recsys-project/data/processed/
├── reviews/                       # отзывы, партиционированы по году
│   ├── amazon_fashion/
│   │   ├── year=2002/…parquet
│   │   ├── …
│   │   ├── year=2023/…parquet
│   │   └── _SUCCESS
│   └── grocery_and_gourmet_food/
└── meta/                          # метаданные товаров, без партиций
    ├── amazon_fashion/
    │   ├── part-00000.parquet
    │   ├── …
    │   └── _SUCCESS
    └── grocery_and_gourmet_food/
```

Пути руками не пишем — их собирает `dataset_uri`:

```python
from recsys.data.paths import dataset_uri

dataset_uri("reviews", "Amazon_Fashion")
# 's3://hse-recsys-project/data/processed/reviews/amazon_fashion'
```

Категория пишется так же, как в названиях файлов на Hugging Face: `Amazon_Fashion`, `Grocery_and_Gourmet_Food`.

## Доступ

1. Положите в `.env` в корне репозитория свои ключи:

   ```bash
   AWS_ACCESS_KEY_ID=...
   AWS_SECRET_ACCESS_KEY=...
   ```

   Эндпоинт, регион и бакет общие и уже прописаны в `src/recsys/settings.py`.

2. В ноутбуке загрузите `.env` первой ячейкой:

   ```python
   %load_ext dotenv
   %dotenv
   ```

## Чтение

### pandas

```python
import pandas as pd
from recsys.data.paths import dataset_uri
from recsys.data.s3 import s3_opts

reviews = pd.read_parquet(
    dataset_uri("reviews", "Amazon_Fashion"),
    columns=["user_id", "parent_asin", "rating", "timestamp", "year"],
    filters=[("year", ">=", 2020)],
    storage_options=s3_opts(),
)

meta = pd.read_parquet(dataset_uri("meta", "Amazon_Fashion"), storage_options=s3_opts())
```

**Фильтруйте при чтении, а не после.** `filters` по `year` читает только нужные папки (partition pruning), а `columns` — только нужные колонки. `df[df.year >= 2020]` после загрузки сначала скачает всё.

- `year` — целое число: `("year", ">=", 2020)`, не `"2020"`.
- Фильтр по `timestamp` тоже ускоряет чтение (Parquet пропускает блоки строк по min/max), но папки целиком не отсекает. Лучше комбинировать: `[("year", "==", 2023), ("timestamp", ">=", pd.Timestamp("2023-06-01"))]`.

### Polars

Удобно для больших категорий: `scan_parquet` ленивый и сам проталкивает фильтры в чтение.

```python
import polars as pl
from recsys.data.paths import dataset_uri
from recsys.settings import get_settings

s = get_settings()
reviews = (
    pl.scan_parquet(
        f"{dataset_uri('reviews', 'Grocery_and_Gourmet_Food')}/**/*.parquet",
        hive_partitioning=True,
        storage_options={"aws_endpoint_url": s.aws_endpoint_url, "aws_region": s.aws_region},
    )
    .filter(pl.col("year") >= 2020)
    .select("user_id", "parent_asin", "rating", "timestamp")
    .collect()
)
```

У Polars свои названия опций, поэтому `s3_opts()` здесь не подходит.

### pyarrow.dataset

Нужен отдельный объект файловой системы S3; пути передаются без `s3://`.

```python
import pyarrow.dataset as ds
from pyarrow import fs
from recsys.data.paths import dataset_uri
from recsys.settings import get_settings

s = get_settings()
s3 = fs.S3FileSystem(
    endpoint_override=s.aws_endpoint_url.removeprefix("https://"),
    region=s.aws_region,
)
dataset = ds.dataset(
    dataset_uri("reviews", "Grocery_and_Gourmet_Food").removeprefix("s3://"),
    filesystem=s3,
    format="parquet",
    partitioning="hive",
)

dataset.schema  # типы колонок без чтения данных

# по батчам, если целиком в память не влезает
for batch in dataset.to_batches(filter=ds.field("year") == 2023, batch_size=500_000):
    ...
```

### Если не хватает памяти

Grocery целиком в pandas, скорее всего, не поместится. Берите только нужные колонки и годы, используйте Polars (`scan_parquet`) или pyarrow по батчам.

## Колонки

### reviews

| Колонка | Тип | Описание |
|---|---|---|
| `user_id` | string | id пользователя |
| `parent_asin` | string | id товара — **используйте его как item id** |
| `asin` | string | id конкретного варианта товара (размер, цвет) |
| `rating` | int8 | оценка 1–5 |
| `title`, `text` | string | заголовок и текст отзыва |
| `timestamp` | timestamp[ms] | время отзыва; без часового пояса, значения в UTC |
| `helpful_vote` | int32 | сколько людей отметили отзыв полезным |
| `verified_purchase` | bool | подтверждённая покупка |
| `n_images` | int16 | сколько фото приложено к отзыву (сами фото не сохраняем) |
| `year` | int32 | год `timestamp`, колонка партиционирования |

Почему `parent_asin`: метаданные (описания, картинки) хранятся по `parent_asin`, а варианты одного товара — это не разные предпочтения. По `asin` товары выглядят холоднее, чем есть.

### meta

Одна строка на `parent_asin`. Точная схема — `META_SCHEMA` в [`datasets.py`](datasets.py).

| Колонка | Тип | Описание |
|---|---|---|
| `parent_asin` | string | id товара, ключ для join с отзывами |
| `title`, `store`, `main_category` | string | название, магазин, категория |
| `price` | float | цена, `NaN` если нет |
| `average_rating`, `rating_number` | float, int | средняя оценка и число оценок на Amazon |
| `features`, `description`, `categories` | list[string] | списки, бывают пустыми |
| `images`, `videos`, `details`, `bought_together` | string (JSON) | вложенные поля, хранятся как JSON-строка |

JSON-поля разбираются так:

```python
import json

images = json.loads(meta.loc[0, "images"])  # список словарей с URL картинок
details = json.loads(meta.loc[0, "details"])  # словарь характеристик, ключи у товаров разные
```

## Соглашения

- **`_SUCCESS` значит, что датасет записан полностью.** Если его нет — данные не готовы.
- **Папку `_staging/` в корне бакета не трогаем** — там лежат незавершённые запуски ETL.
- **В `processed/` руками не пишем.** Всё там собирается скриптом `etl.py`; файлы, начинающиеся с `_`, читатели игнорируют.
- **Исходники — на Hugging Face** ([McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)), сырые JSONL в S3 не храним.

## Пересборка датасета

Нужна редко. Запуск из корня репозитория:

```bash
uv run python src/recsys/data/etl.py --kind reviews --category Amazon_Fashion --dry-run
uv run python src/recsys/data/etl.py --kind reviews --category Amazon_Fashion --overwrite
```

`--overwrite` заменяет старую версию только после успешного завершения. Если запуск упал жёстко (закрыли ноутбук, kill -9), мусор чистит `uv run python src/recsys/data/cleanup.py`.
