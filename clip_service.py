"""
Vision-Language Embedding Service
=================================
Separate process that consumes the ``person_crops`` Kafka topic, encodes each
crop with a configurable OpenCLIP model, and upserts the resulting vector +
metadata into Qdrant.

The model is selected via config.json:

    "model": {
        "name": "ViT-SO400M-14-SigLIP-384",
        "pretrained": "webli"
    }

Run:
    python3 clip_service.py --config config.json
"""

import argparse
import base64
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List

import cv2
import numpy as np
import open_clip
import torch
from kafka import KafkaConsumer
from kafka.errors import KafkaError
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("clip_service")


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


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------

def ensure_collection(client: QdrantClient, collection: str, dim: int) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if collection in existing:
        info = client.get_collection(collection)
        existing_dim = info.config.params.vectors.size
        if existing_dim != dim:
            logger.warning(
                "Collection '%s' has dim %d but model produces %d — recreating.",
                collection, existing_dim, dim,
            )
            client.delete_collection(collection)
        else:
            logger.info("Collection '%s' exists (dim=%d)", collection, existing_dim)
            return

    logger.info("Creating collection '%s' (dim=%d)", collection, dim)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    client.create_payload_index(collection, "sensor_id",  "keyword")
    client.create_payload_index(collection, "tracker_id", "integer")
    client.create_payload_index(collection, "pad_index",  "integer")
    client.create_payload_index(collection, "timestamp",  "datetime")
    logger.info("Collection and payload indexes created")


# ---------------------------------------------------------------------------
# Model wrapper — works with any OpenCLIP model
# ---------------------------------------------------------------------------

class VisionEncoder:
    def __init__(self, model_name: str, pretrained: str, device: str):
        logger.info("Loading model '%s' (pretrained=%s) on %s", model_name, pretrained, device)
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

        # Auto-detect embedding dimension from a dummy forward pass
        # Infer expected input size from model's visual config
        img_size = getattr(self.model.visual, 'image_size', None)
        if isinstance(img_size, (tuple, list)):
            h, w = img_size
        else:
            h = w = img_size or 224
        with torch.no_grad():
            dummy = torch.randn(1, 3, h, w).to(device)
            self.embed_dim = self.model.encode_image(dummy).shape[1]

        logger.info("Model ready — embedding dim=%d", self.embed_dim)

    @torch.no_grad()
    def encode_images(self, pil_images: List[Image.Image]) -> np.ndarray:
        tensors = torch.stack(
            [self.preprocess(img) for img in pil_images]
        ).to(self.device)
        embeddings = self.model.encode_image(tensors)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().float().numpy()

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        emb = self.model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().float().numpy()[0]


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def decode_crop(b64_str: str):
    try:
        jpeg_bytes = base64.b64decode(b64_str)
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    except Exception as e:
        logger.warning("Failed to decode crop: %s", e)
        return None


def process_batch(
    messages: List[Dict],
    encoder: VisionEncoder,
    qdrant: QdrantClient,
    collection: str,
) -> None:
    pil_images: List[Image.Image] = []
    valid: List[Dict] = []

    for msg in messages:
        img = decode_crop(msg.get("crop_jpeg_b64", ""))
        if img is not None:
            pil_images.append(img)
            valid.append(msg)

    if not pil_images:
        return

    embeddings = encoder.encode_images(pil_images)

    points: List[PointStruct] = []
    for emb, msg in zip(embeddings, valid):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=emb.tolist(),
            payload={
                "sensor_id":    msg.get("sensor_id", ""),
                "tracker_id":   msg.get("tracker_id", -1),
                "confidence":   msg.get("confidence", 0.0),
                "bbox":         msg.get("bbox", []),
                "frame_number": msg.get("frame_number", 0),
                "pad_index":    msg.get("pad_index", 0),
                "timestamp":    msg.get(
                    "timestamp",
                    datetime.now(timezone.utc).isoformat()
                ),
            },
        ))

    try:
        qdrant.upsert(collection_name=collection, points=points)
        logger.info("Upserted %d vectors to Qdrant", len(points))
    except Exception as e:
        logger.error("Qdrant upsert failed: %s", e)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(config: dict) -> None:
    kafka_cfg  = config["kafka"]
    qdrant_cfg = config["qdrant"]
    model_cfg  = config.get("model", {})

    model_name = model_cfg.get("name", "ViT-B-32")
    pretrained = model_cfg.get("pretrained", "openai")
    collection = qdrant_cfg.get("collection_name", "persons")

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = VisionEncoder(model_name, pretrained, device)

    qdrant = _qdrant_client(qdrant_cfg)
    ensure_collection(qdrant, collection, encoder.embed_dim)

    batch_size    = int(config.get("batch_size", 32))
    batch_timeout = float(config.get("batch_timeout_ms", 200)) / 1000.0

    consumer = None
    while True:
        try:
            if consumer is None:
                consumer = KafkaConsumer(
                    kafka_cfg["topic"],
                    bootstrap_servers=kafka_cfg["bootstrap_servers"],
                    group_id=kafka_cfg.get("group_id", "clip_embedding_service"),
                    auto_offset_reset="latest",
                    enable_auto_commit=True,
                    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                    max_poll_records=batch_size,
                    fetch_max_bytes=52428800,
                )
                logger.info(
                    "Connected to Kafka %s, consuming topic '%s'",
                    kafka_cfg["bootstrap_servers"],
                    kafka_cfg["topic"],
                )

            raw = consumer.poll(timeout_ms=int(batch_timeout * 1000))
            batch: List[Dict] = []
            for records in raw.values():
                for record in records:
                    batch.append(record.value)

            if batch:
                process_batch(batch, encoder, qdrant, collection)

        except KafkaError as e:
            logger.error("Kafka error: %s — reconnecting in 15 s", e)
            consumer = None
            time.sleep(15)
        except Exception as e:
            logger.error("Unexpected error: %s — restarting in 10 s", e)
            consumer = None
            time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision-Language Embedding Service")
    parser.add_argument("--config", default="config.json", help="Path to config JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    run(cfg)
