"""Import official UMLS MRCONSO.RRF into umls_test.umls_strings_raw_test.

This script implements the simplified public path for the UMLS string table
used by QKG's UMLS-backed entity fallback.
"""

from pymongo import MongoClient, UpdateOne

from lib.runtime_config import get_mongo_uri, get_path_config

MONGO_URI = get_mongo_uri("umls")
INPUT_FILE = get_path_config("umls_mrconso_rrf")
DB_NAME = "umls_test"
COLLECTION = "umls_strings_raw_test"
BATCH_SIZE = 5000

COLUMNS = [
    "cui",
    "language",
    "term_status",
    "lui",
    "string_type",
    "string_identifier",
    "is_preferred",
    "aui",
    "source_aui",
    "source_cui",
    "source_descriptor_dui",
    "source",
    "source_term_type",
    "source_code",
    "source_name",
    "unused1",
    "unused2",
    "unused3",
    "unused4",
]


def parse_rrf_line(line: str) -> dict | None:
    parts = line.rstrip("\n").split("|")
    if parts and parts[-1] == "":
        parts = parts[:-1]

    if len(parts) < len(COLUMNS):
        parts.extend([""] * (len(COLUMNS) - len(parts)))
    elif len(parts) > len(COLUMNS):
        parts = parts[: len(COLUMNS)]

    doc = dict(zip(COLUMNS, parts))
    if doc.get("language") != "ENG":
        return None
    return doc


def flush_batch(col, batch):
    if not batch:
        return 0
    col.bulk_write(batch, ordered=False)
    count = len(batch)
    batch.clear()
    return count


def main():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    col = db[COLLECTION]

    if COLLECTION in db.list_collection_names():
        print(f"Dropping existing '{COLLECTION}' collection...")
        db.drop_collection(COLLECTION)
        col = db[COLLECTION]

    batch = []
    total = 0

    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            doc = parse_rrf_line(line)
            if doc is None:
                continue

            sid = doc.pop("string_identifier")
            batch.append(UpdateOne({"_id": sid}, {"$set": doc}, upsert=True))

            if len(batch) >= BATCH_SIZE:
                total += flush_batch(col, batch)
                if total % 100000 == 0:
                    print(f"  Inserted {total:,} documents...")

            if line_no % 1000000 == 0:
                print(f"  Processed {line_no:,} source rows...")

    total += flush_batch(col, batch)

    print(f"Done. Inserted {total:,} English UMLS strings into {DB_NAME}.{COLLECTION}")

    print("Creating indexes...")
    col.create_index("source_name", name="source_name_1")
    col.create_index("cui", name="cui_1")
    col.create_index([("source", 1), ("source_code", 1)], name="source_1_source_code_1")
    print("Indexes created.")


if __name__ == "__main__":
    main()
