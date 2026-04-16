"""
Import PrimeKg.csv into MongoDB database 'primeKG', collection 'relations'.
"""
import csv
import time
from pymongo import MongoClient, InsertOne
from lib.runtime_config import get_mongo_uri, get_path_config

MONGO_URI = get_mongo_uri("primekg")
CSV_FILE = get_path_config("primekg_csv")
BATCH_SIZE = 10000


def main():
    client = MongoClient(MONGO_URI)
    db = client["primeKG"]

    # Drop existing collection if any
    if "relations" in db.list_collection_names():
        print("Dropping existing 'relations' collection...")
        db.drop_collection("relations")

    col = db["relations"]

    print(f"Reading {CSV_FILE}...")
    start = time.time()

    batch = []
    total = 0

    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            doc = {
                "relation": row["relation"],
                "display_relation": row["display_relation"],
                "x_index": int(row["x_index"]),
                "x_id": row["x_id"],
                "x_type": row["x_type"],
                "x_name": row["x_name"],
                "x_source": row["x_source"],
                "y_index": int(row["y_index"]),
                "y_id": row["y_id"],
                "y_type": row["y_type"],
                "y_name": row["y_name"],
                "y_source": row["y_source"],
            }

            batch.append(InsertOne(doc))
            total += 1

            if len(batch) >= BATCH_SIZE:
                col.bulk_write(batch, ordered=False)
                batch = []
                if total % 500000 == 0:
                    elapsed = time.time() - start
                    print(f"  {total:,} rows inserted ({elapsed:.1f}s)")

        # Final batch
        if batch:
            col.bulk_write(batch, ordered=False)

    elapsed = time.time() - start
    print(f"\nDone. Inserted {total:,} documents in {elapsed:.1f}s")

    # Create indexes
    print("Creating indexes...")
    col.create_index([("x_source", 1), ("x_id", 1)])
    col.create_index([("y_source", 1), ("y_id", 1)])
    col.create_index([("x_index", 1)])
    col.create_index([("y_index", 1)])
    col.create_index([("relation", 1)])
    col.create_index([("x_name", 1)])
    col.create_index([("y_name", 1)])
    print("Indexes created.")

    # Verify
    count = col.count_documents({})
    print(f"Verification: {count:,} documents in primeKG.relations")


if __name__ == "__main__":
    main()
