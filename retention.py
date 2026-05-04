"""
Qdrant Retention Script
=======================
Runs hourly and deletes vectors older than RETENTION_DAYS from the Qdrant
persons collection.

Run:
    python3 retention.py --config config.json

Optional overrides:
    python3 retention.py --config config.json --retention-days 14 --interval-hours 6
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone

from qdrant_client import QdrantClient
from qdrant_client.models import DatetimeRange, FieldCondition, Filter

# ---------------------------------------------------------------------------
# Configurable defaults
# ---------------------------------------------------------------------------

RETENTION_DAYS = 7       # keep this many days of data
INTERVAL_HOURS = 1       # how often to run the cleanup

# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("retention")


def _qdrant_client(qdrant_cfg: dict) -> QdrantClient:
    kw = dict(
        host=qdrant_cfg["host"],
        port=int(qdrant_cfg["port"]),
        https=bool(qdrant_cfg.get("https", False)),
        check_compatibility=bool(qdrant_cfg.get("check_compatibility", False)),
    )
    key = qdrant_cfg.get("api_key") or os.environ.get("QDRANT_API_KEY")
    if key:
        kw["api_key"] = str(key).strip()
    return QdrantClient(**kw)


def delete_old_points(client: QdrantClient, collection: str, retention_days: int) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)

    before_count = client.count(collection_name=collection, exact=True).count

    client.delete(
        collection_name=collection,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="timestamp",
                    range=DatetimeRange(lt=cutoff),
                )
            ]
        ),
    )

    after_count = client.count(collection_name=collection, exact=True).count
    deleted = before_count - after_count
    return deleted


def run(config: dict, retention_days: int, interval_hours: int) -> None:
    qdrant_cfg = config["qdrant"]
    collection = qdrant_cfg.get("collection_name", "persons")

    client = _qdrant_client(qdrant_cfg)
    logger.info(
        "Retention service started — collection='%s', retention=%d days, interval=%d h",
        collection, retention_days, interval_hours,
    )

    interval_seconds = interval_hours * 3600

    while True:
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
            logger.info("Running retention check — deleting points older than %s", cutoff.isoformat())

            deleted = delete_old_points(client, collection, retention_days)

            if deleted > 0:
                logger.info("Deleted %d points older than %d days", deleted, retention_days)
            else:
                logger.info("No expired points found")

        except Exception as e:
            logger.error("Retention check failed: %s", e)

        logger.info("Next check in %d hour(s)", interval_hours)
        time.sleep(interval_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qdrant retention cleanup service")
    parser.add_argument("--config", default="config.json", help="Path to config JSON")
    parser.add_argument(
        "--retention-days",
        type=int,
        default=RETENTION_DAYS,
        help=f"Days of data to keep (default: {RETENTION_DAYS})",
    )
    parser.add_argument(
        "--interval-hours",
        type=int,
        default=INTERVAL_HOURS,
        help=f"Hours between cleanup runs (default: {INTERVAL_HOURS})",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    run(cfg, args.retention_days, args.interval_hours)
