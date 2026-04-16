"""Export primeKG.entities from MongoDB to JSONL for public release."""

from __future__ import annotations

import json

from pymongo import MongoClient

from lib.runtime_config import get_mongo_uri, get_path_config


MONGO_URI = get_mongo_uri("primekg")
OUTPUT_FILE = get_path_config("primekg_entities_jsonl")
DB_NAME = "primeKG"
COLLECTION = "entities"

FIELDS = [
    "index",
    "id",
    "type",
    "name",
    "source",
    "cui",
    "cui_score",
    "cui_method",
]


def main():
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION]

    total = 0
    print(f"Exporting {DB_NAME}.{COLLECTION} to {OUTPUT_FILE}...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as handle:
        cursor = col.find({}, {field: 1 for field in FIELDS}).sort("index", 1)
        for doc in cursor:
            out = {field: doc.get(field) for field in FIELDS}
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            total += 1
            if total % 10000 == 0:
                print(f"  Exported {total:,} documents...")

    print(f"Done. Exported {total:,} documents.")


if __name__ == "__main__":
    main()
