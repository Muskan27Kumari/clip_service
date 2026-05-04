import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
_venv_py = _root / "venv" / "bin" / "python"
if (
    __name__ == "__main__"
    and _venv_py.is_file()
    and Path(sys.executable).resolve() != _venv_py.resolve()
    and str(os.environ.get("CLIP_SERVICE_USE_VENV", "1")).lower() not in ("0", "false", "no")
):
    os.execv(str(_venv_py), [str(_venv_py), str(_root / "video_streamer.py"), *sys.argv[1:]])

import argparse
import base64
import json
import logging
import socket
import threading
import time
import urllib.request
from datetime import datetime, timezone

import cv2

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

from flask import Flask, request, Response
from kafka import KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# OpenCV project sample: pedestrians (≈8 MiB). Used when no webcam (--fetch-sample).
SAMPLE_VIDEO_NAME = "vtest.avi"
SAMPLE_VIDEO_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _parse_video_source_token(token: str) -> int | str:
    token = token.strip()
    try:
        return int(token)
    except ValueError:
        return token


def _find_first_camera_index(*, max_index: int = 10) -> int | None:
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        opened = cap.isOpened()
        cap.release()
        if opened:
            return i
    return None


def list_camera_indices(*, max_index: int = 10) -> None:
    print(f"Camera indices 0–{max_index} (opened / readable one frame):")
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        opened = cap.isOpened()
        readable = False
        if opened:
            readable, _ = cap.read()
        cap.release()
        tag = "ok" if (opened and readable) else ("open" if opened else "—")
        print(f"  {i}: {tag}")


def fetch_sample_video(dest: Path | None = None) -> Path:
    dest = dest or (_repo_root() / "data" / SAMPLE_VIDEO_NAME)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        logger.info("Sample already present at %s", dest)
        return dest
    logger.info("Downloading sample video (~8 MiB) from OpenCV…")
    req = urllib.request.Request(SAMPLE_VIDEO_URL, headers={"User-Agent": "clip_service/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    logger.info("Wrote %s", dest)
    return dest


def resolve_video_source(cli_value: str) -> int | str | None:
    """
    Resolve --video-source.
    ``auto``: CLIP_VIDEO_SOURCE env, else first working camera index, else data/vtest.avi if present.
    """
    if cli_value != "auto":
        return _parse_video_source_token(cli_value)

    env = os.environ.get("CLIP_VIDEO_SOURCE", "").strip()
    if env:
        logger.info("Using CLIP_VIDEO_SOURCE=%r", env)
        return _parse_video_source_token(env)

    cam = _find_first_camera_index()
    if cam is not None:
        logger.info("Using first available camera (index %d)", cam)
        return cam

    sample = _repo_root() / "data" / SAMPLE_VIDEO_NAME
    if sample.is_file():
        logger.info("No camera found; using sample file %s", sample)
        return str(sample)

    logger.error(
        "No webcam found and %s is missing. Options:\n"
        "  • Try another index:  python video_streamer.py --video-source 1\n"
        "  • List devices:       python video_streamer.py --list-cameras\n"
        "  • Download sample:    python video_streamer.py --fetch-sample\n"
        "  • Use your own file:  python video_streamer.py --video-source /path/to/video.mp4\n"
        "Linux: if you have a camera, check /dev/video* and that your user is in the ``video`` group.",
        sample,
    )
    return None


# --- Snapshot Server Logic ---
app = Flask(__name__)
frame_cache = {}  # {timestamp: frame} — only frames with a person detection, last MAX_CACHE_SIZE
cache_lock = threading.Lock()
MAX_CACHE_SIZE = 100

_SNAPSHOT_MISS = (
    "Snapshot not found. Full frames are cached only for recent person detections "
    "while this process is running (about the last %d frames in RAM). "
    "Search results from Qdrant can be older; those timestamps are not kept here "
    "after eviction or restart. Run the streamer, produce new hits, and open "
    "snapshot right away, or rely on bbox metadata without a full-frame image."
) % MAX_CACHE_SIZE


def _parse_iso_utc(s: str):
    if not s or not isinstance(s, str):
        return None
    try:
        s2 = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s2)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _lookup_frame(timestamp: str):
    """Exact key match, then nearest cached frame within 0.5s (Qdrant vs. streamer string quirks)."""
    with cache_lock:
        if timestamp in frame_cache:
            return frame_cache[timestamp]
        req_dt = _parse_iso_utc(timestamp)
        if req_dt is None:
            return None
        best_key = None
        best_delta = None
        for k in frame_cache:
            kd = _parse_iso_utc(k)
            if kd is None:
                continue
            delta = abs((kd - req_dt).total_seconds())
            if delta <= 0.5 and (best_delta is None or delta < best_delta):
                best_delta = delta
                best_key = k
        if best_key is not None:
            return frame_cache[best_key]
    return None


@app.route('/snapshot')
def get_snapshot():
    # `camera` is accepted for API parity with the UI; cache is single-stream for now.
    timestamp = request.args.get('timestamp')

    frame = _lookup_frame(timestamp) if timestamp else None

    if frame is not None:
        _, buffer = cv2.imencode('.jpg', frame)
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    return _SNAPSHOT_MISS, 404, {"Content-Type": "text/plain; charset=utf-8"}

def start_snapshot_server(port=8009):
    logger.info(f"Starting snapshot server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

# --- Streamer Logic ---

def _load_config(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        return {}
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def _kafka_settings(config: dict) -> tuple[str, str]:
    k = config.get("kafka") or {}
    bootstrap = k.get("bootstrap_servers", "localhost:9092")
    topic = k.get("topic", "person_crops")
    return bootstrap, topic


def _silence_kafka_client_logging() -> None:
    """kafka-python logs every socket retry at INFO; that drowns real messages."""
    for name in ("kafka", "kafka.conn", "kafka.client", "kafka.producer", "kafka.cluster"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _first_bootstrap_host_port(bootstrap_servers: str | list | tuple) -> tuple[str, int] | None:
    if isinstance(bootstrap_servers, (list, tuple)):
        token = str(bootstrap_servers[0]).strip() if bootstrap_servers else ""
    else:
        token = str(bootstrap_servers).split(",")[0].strip()
    if not token or ":" not in token:
        return None
    host, _, port_s = token.rpartition(":")
    if host.startswith("["):
        return None
    try:
        return host, int(port_s)
    except ValueError:
        return None


def _tcp_connect_ok(host: str, port: int, *, timeout_s: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


class _NullKafkaProducer:
    """Drop-in when running with --no-kafka (snapshot + YOLO only)."""

    def send(self, *args, **kwargs):
        class _Future:
            def add_errback(self, *_a, **_k):
                return self

            def add_callback(self, *_a, **_k):
                return self

            def get(self, *_a, **_k):
                return None

        return _Future()

    def flush(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass


def _connect_kafka_producer(bootstrap_servers: str | list | tuple, *, max_wait_s: float = 300.0):
    """
    Block until Kafka accepts a producer (or timeout).
    Uses a TCP reachability check first so kafka-python does not spam INFO on every retry.
    """
    _silence_kafka_client_logging()
    deadline = time.monotonic() + max_wait_s
    ep = _first_bootstrap_host_port(bootstrap_servers)
    backoff_s = 2.0
    attempt = 0
    last_warn = 0.0

    logger.info(
        "Waiting for Kafka at %r (up to %.0fs). Start broker: docker compose up -d kafka",
        bootstrap_servers,
        max_wait_s,
    )

    while time.monotonic() < deadline:
        attempt += 1
        if ep is not None and not _tcp_connect_ok(ep[0], ep[1], timeout_s=1.0):
            now = time.monotonic()
            if now - last_warn >= 20.0 or attempt == 1:
                logger.warning(
                    "Kafka not listening at %s:%s yet (nothing on that port). "
                    "Typical fix: docker compose up -d kafka",
                    ep[0],
                    ep[1],
                )
                last_warn = now
            time.sleep(min(backoff_s, 30.0))
            backoff_s = min(backoff_s * 1.4, 30.0)
            continue

        producer = None
        try:
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                retries=2,
                request_timeout_ms=12_000,
            )
            if getattr(producer, "bootstrap_connected", lambda: True)():
                logger.info("Connected to Kafka at %s", bootstrap_servers)
                return producer
            producer.close()
        except (NoBrokersAvailable, KafkaError, OSError, ConnectionError) as e:
            now = time.monotonic()
            if now - last_warn >= 20.0 or attempt == 1:
                logger.warning(
                    "Kafka client could not finish handshake at %r (%s). Retrying…",
                    bootstrap_servers,
                    e,
                )
                last_warn = now
            if producer is not None:
                try:
                    producer.close()
                except Exception:
                    pass
        except Exception as e:
            logger.warning("Kafka connection error (%s): %s", type(e).__name__, e)
            if producer is not None:
                try:
                    producer.close()
                except Exception:
                    pass
        time.sleep(min(backoff_s, 10.0))
        backoff_s = min(backoff_s * 1.25, 30.0)

    raise RuntimeError(
        f"Could not reach Kafka at {bootstrap_servers!r} within {max_wait_s:.0f}s. "
        "Run: docker compose up -d kafka   (or use --no-kafka to test video only)"
    )


def run_video_streamer(
    video_source="video.mp4",
    kafka_bootstrap: str = "localhost:9092",
    kafka_topic: str = "person_crops",
    sensor_id: str = "office-cam01",
    *,
    no_kafka: bool = False,
):
    # Start snapshot server in background
    server_thread = threading.Thread(target=start_snapshot_server, daemon=True)
    server_thread.start()

    # 1. Load YOLOv8 model
    logger.info("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")

    # 2. Setup Kafka Producer
    if no_kafka:
        _silence_kafka_client_logging()
        logger.warning(
            "Running with --no-kafka: snapshots and YOLO run, but person_crops are not sent to Kafka."
        )
        producer = _NullKafkaProducer()
    else:
        logger.info("Connecting to Kafka at %s (topic=%s)...", kafka_bootstrap, kafka_topic)
        try:
            producer = _connect_kafka_producer(kafka_bootstrap)
        except RuntimeError as e:
            logger.error("%s", e)
            return

    # 3. Open Video Source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error(
            "Could not open video source %r. Try --list-cameras, another --video-source index, "
            "or --fetch-sample for an offline file.",
            video_source,
        )
        producer.flush()
        producer.close()
        return

    logger.info("Starting processing. Press Ctrl+C to stop.")
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream or error reading frame.")
                break

            frame_count += 1
            ts = datetime.now(timezone.utc).isoformat()
            
            # Run inference
            results = model(frame, verbose=False)

            detected_person = False
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    if cls == 0:  # Person
                        detected_person = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0: continue

                        _, buffer = cv2.imencode('.jpg', crop)
                        img_b64 = base64.b64encode(buffer).decode('utf-8')

                        message = {
                            "sensor_id": sensor_id,
                            "tracker_id": 1,
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2],
                            "frame_number": frame_count,
                            "pad_index": 0,
                            "crop_jpeg_b64": img_b64,
                            "timestamp": ts
                        }
                        producer.send(kafka_topic, message)

            if detected_person:
                with cache_lock:
                    frame_cache[ts] = frame.copy()
                    if len(frame_cache) > MAX_CACHE_SIZE:
                        oldest_ts = next(iter(frame_cache))
                        del frame_cache[oldest_ts]

            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count} frames...")
                producer.flush()

    except KeyboardInterrupt:
        logger.info("Stopping streamer...")
    finally:
        cap.release()
        producer.flush()
        producer.close()
        logger.info("Resources released.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO + Kafka person crop streamer and snapshot server.")
    parser.add_argument(
        "--config",
        default="config.json",
        help="JSON config (same shape as clip_service); kafka.bootstrap_servers and kafka.topic are used.",
    )
    parser.add_argument(
        "--video-source",
        default="auto",
        help="OpenCV source: 'auto' (env CLIP_VIDEO_SOURCE, else first camera, else data/vtest.avi), "
        "integer camera index, file path, or URL.",
    )
    parser.add_argument("--sensor-id", default="office_cam_01")
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Print camera indices 0–10 that open (and read one frame), then exit.",
    )
    parser.add_argument(
        "--fetch-sample",
        action="store_true",
        help="Download OpenCV vtest.avi sample (~8 MiB) to data/vtest.avi and exit.",
    )
    parser.add_argument(
        "--no-kafka",
        action="store_true",
        help="Do not connect to Kafka; run YOLO + snapshot server only (no person_crops messages).",
    )
    args = parser.parse_args()

    if args.list_cameras:
        list_camera_indices()
        sys.exit(0)
    if args.fetch_sample:
        fetch_sample_video()
        sys.exit(0)

    cfg = _load_config(args.config)
    bootstrap, topic = _kafka_settings(cfg)

    video_src = resolve_video_source(args.video_source)
    if video_src is None:
        sys.exit(1)

    run_video_streamer(
        video_source=video_src,
        kafka_bootstrap=bootstrap,
        kafka_topic=topic,
        sensor_id=args.sensor_id,
        no_kafka=args.no_kafka,
    )
