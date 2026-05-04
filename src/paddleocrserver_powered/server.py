"""
paddleocrserver_powered.server — The server that reads what nothing else can read.

A neural network trained on millions of text images from around the world,
deployed locally, called over HTTP from any client, to read plain text
on legacy interfaces that don't even know what a DOM is.

Baidu spent years building PaddleOCR.
We installed it with `pip install paddleocr`.
One of the best command lines we've ever typed.

Architecture:
    Any client (Java, Python, RF, curl) → HTTP POST localhost:5000 → Flask → PaddleOCR → JSON

Why Flask and not FastAPI?
    Because FastAPI would have required explaining asyncio to someone.
    Flask starts, responds, does its job, doesn't complain.
    Exactly what we ask from everyone on this project.

Why Waitress and not Flask's dev server?
    Because the dev server kept saying "WARNING: Do not use in production"
    and we eventually decided to listen. Probably the only time we followed
    a warning without waiting for something to explode first.

Why threads=1 in Waitress?
    Because PaddleOCR is not thread-safe.
    Because we tried threads=4.
    Because we don't talk about it anymore.

Multi-language?
    Each enabled language has its own PaddleOCR engine instance, picked from
    a (lang, mode) cache. Engines for languages declared in languages.yaml
    are eager-loaded at startup in 'fast' mode (no cold start). The 'full'
    pipeline (doc orientation classify + textline orientation + dewarping)
    is loaded lazily the first time someone asks for preprocess="full".

Author: Julien MER
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import socket
import sys
import time
from importlib import resources
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import yaml
from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from paddleocr import PaddleOCR
from waitress import serve

PACKAGE_NAME = "paddleocrserver_powered"

# Languages we ship YAMLs for. Adding a new one is mechanical: drop two YAMLs
# (ocr_config_<code>.yaml + ocr_config_<code>_full.yaml) next to the others
# and append the code here.
SUPPORTED_LANGUAGES = (
    "latin",
    "japan",
    "korean",
    "chinese_cht",
    "cyrillic",
    "arabic",
    "devanagari",
)


def is_port_in_use(host: str, port: int) -> bool:
    """
    Checks if the port is already taken before starting the server.

    Logic: tries to bind a socket on host:port. If OSError, port is taken.

    Exists because one day two instances ran simultaneously, each convinced
    it was alone, producing random OCR results depending on which one answered
    first. It was fascinating to debug. It was also a Friday evening.
    This function exists so it never happens again.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


# ----------------------------
# Configuration via env vars (12-factor style, zero hardcoded values)
# ----------------------------
DEFAULT_RATE_LIMIT = os.getenv("OCR_RATE_LIMIT_DEFAULT", "100 per minute")
OCR_RATE_LIMIT = os.getenv("OCR_RATE_LIMIT_OCR", "30 per minute")
RATE_LIMIT_STORAGE_URI = os.getenv("OCR_RATE_LIMIT_STORAGE", "memory://")
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
# .bmp is here for legacy SikuliX screen captures. It stays there forever.


# ----------------------------
# Logging — clean format, stdout only, nothing fancy
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("SecurePaddleOCR")


# ----------------------------
# Flask app + rate limiter + security headers
# ----------------------------
app = Flask(__name__)
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[DEFAULT_RATE_LIMIT],
    storage_uri=RATE_LIMIT_STORAGE_URI,
    strategy="moving-window"
)


@app.after_request
def set_secure_headers(resp):
    """
    Adds standard security headers to every response.

    Yes, this runs on localhost. Yes, it's overkill.
    No, we're not removing it, because the day we put this behind
    a reverse proxy, we want zero code changes.
    """
    resp.headers["Server"] = "SecurePaddleOCR"
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["X-XSS-Protection"] = "1; mode=block"
    return resp


# ----------------------------
# Engine cache + language config
# ----------------------------
# ENGINES[(lang, mode)] -> PaddleOCR instance. Built up lazily, except for
# the (enabled_lang, "fast") pairs which are eager-loaded at boot.
ENGINES: dict[tuple[str, str], PaddleOCR] = {}
ENGINES_LOCK = Lock()
ENABLED_LANGUAGES: list[str] = []
DEFAULT_LANGUAGE: str = "latin"
start_time: float = 0.0


def _resource_path(filename: str) -> Path:
    """Resolve a packaged resource (works in installed wheels and editable installs)."""
    return Path(str(resources.files(PACKAGE_NAME).joinpath(filename)))


def resolve_languages_config(cli_path: str | None = None) -> Path:
    """
    Resolution order: CLI flag > OCR_LANGUAGES_CONFIG env var > packaged default.

    The packaged default ships latin=true and everything else=false, so a
    plain `pip install` + `paddleocrserver-powered` Just Works.
    """
    if cli_path:
        return Path(cli_path).expanduser().resolve()
    env_path = os.getenv("OCR_LANGUAGES_CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return _resource_path("languages.yaml")


def load_languages_config(path: Path) -> tuple[list[str], str]:
    """
    Reads languages.yaml and returns (enabled_languages, default_language).

    Validates that:
      - all enabled codes are supported
      - at least one language is enabled (otherwise the server is useless)
      - default_language is in the enabled list (otherwise /ocr would 400 on
        every request that omits the lang param — we'd rather fail at boot)
    """
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    enabled_map = cfg.get("enabled", {}) or {}
    enabled = [lang for lang, on in enabled_map.items() if on]
    default_lang = cfg.get("default_language", "latin")

    unknown = [lang for lang in enabled if lang not in SUPPORTED_LANGUAGES]
    if unknown:
        raise ValueError(
            f"Unknown language(s) in {path}: {unknown}. "
            f"Supported: {list(SUPPORTED_LANGUAGES)}"
        )
    if not enabled:
        raise ValueError(f"No languages enabled in {path}")
    if default_lang not in enabled:
        raise ValueError(
            f"default_language '{default_lang}' is not in enabled list {enabled}"
        )
    return enabled, default_lang


def yaml_for(lang: str, mode: str) -> Path:
    """Map (lang, mode) to the packaged YAML path."""
    suffix = "_full" if mode == "full" else ""
    return _resource_path(f"ocr_config_{lang}{suffix}.yaml")


def get_engine(lang: str, mode: str) -> PaddleOCR:
    """
    Returns a cached PaddleOCR engine for (lang, mode), instantiating it on
    miss. Double-checked locking because two requests can arrive at the same
    millisecond and we are NOT going to load the same 200MB model twice.
    """
    key = (lang, mode)
    engine = ENGINES.get(key)
    if engine is not None:
        return engine
    with ENGINES_LOCK:
        engine = ENGINES.get(key)
        if engine is not None:
            return engine
        cfg_path = yaml_for(lang, mode)
        log.info(f"[LOAD] PaddleOCR engine lang={lang} mode={mode} ({cfg_path.name})")
        t0 = time.time()
        engine = PaddleOCR(paddlex_config=str(cfg_path))
        log.info(f"[LOAD] lang={lang} mode={mode} ready in {time.time() - t0:.2f}s")
        ENGINES[key] = engine
        return engine


# ----------------------------
# Helpers
# ----------------------------
def allowed_file(path: str) -> bool:
    """
    Validates file extension before OCR processing.

    Logic: lowercase the extension, check it's in the whitelist.

    Without this, someone would eventually send a PDF, a .bak file,
    or worse — an .xlsx — to an OCR engine and open a ticket saying
    "it doesn't work".
    """
    ext = os.path.splitext(path)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def _validate_lang_mode(lang: str | None, preprocess: str | None):
    """
    Returns (lang, mode, error_response). If error_response is not None,
    the caller should return it directly. Otherwise (lang, mode) are the
    validated values to use downstream.
    """
    lang = lang or DEFAULT_LANGUAGE
    mode = (preprocess or "fast").lower()
    if lang not in ENABLED_LANGUAGES:
        return None, None, (
            jsonify({
                "success": False,
                "error": f"Language '{lang}' is not enabled.",
                "enabled_languages": ENABLED_LANGUAGES,
                "default_language": DEFAULT_LANGUAGE,
            }),
            400,
        )
    if mode not in ("fast", "full"):
        return None, None, (
            jsonify({
                "success": False,
                "error": f"Invalid preprocess '{mode}', expected 'fast' or 'full'",
            }),
            400,
        )
    return lang, mode, None


def process_ocr(image_path: str, lang: str, mode: str):
    """
    Runs PaddleOCR on an image and returns structured results.

    Logic:
        1. Look up (or lazy-create) the PaddleOCR engine for (lang, mode)
        2. Call engine.predict(path) — PaddleOCR does its neural magic
        3. Extract the JSON structure from results[0]
        4. Walk into result_data['res'] to find rec_texts, rec_scores, rec_polys
        5. Zip them together into a clean [{text, confidence, bbox}, ...] list
        6. Return the list (empty if nothing detected)
        7. Always gc.collect() in finally — PaddleOCR leaks memory silently

    The PaddleOCR output format changed between versions. Specifically,
    it changed exactly the day we finished stabilizing the previous one.
    This function navigates the new structure and reconstructs something
    usable. The [DEBUG] logs are intentional — they've saved several
    evenings and we're keeping them.

    The gc.collect() in finally is empirical, not documented anywhere.
    It costs almost nothing. The memory leak at 3am costs more.

    Returns: list of {"text": str, "confidence": float, "bbox": list}
    Raises: whatever PaddleOCR feels like raising that day
    """
    try:
        log.info(f"[DEBUG] 1. process_ocr start lang={lang} mode={mode}...")
        engine = get_engine(lang, mode)
        results = engine.predict(image_path)
        log.info(f"[DEBUG] 2. predict() OK, type: {type(results)}")

        ocr_results = []
        if results and len(results) > 0:
            log.info("[DEBUG] 3. results[0] exists")
            result_data = results[0].json
            log.info(f"[DEBUG] 4. json extracted, keys: {list(result_data.keys())}")

            if 'res' in result_data:
                res = result_data['res']
                rec_texts = res.get('rec_texts', [])
                rec_scores = res.get('rec_scores', [])
                rec_polys = res.get('rec_polys', [])

                log.info(f"[DEBUG] 5. Found {len(rec_texts)} texts")

                for i, text in enumerate(rec_texts):
                    if i < len(rec_scores) and i < len(rec_polys):
                        bbox_list = rec_polys[i].tolist() if hasattr(rec_polys[i], 'tolist') else rec_polys[i]
                        ocr_results.append({
                            "text": text,
                            "confidence": float(rec_scores[i]),
                            "bbox": bbox_list
                        })
            else:
                log.warning("[DEBUG] 6. No 'res' key found")
        else:
            log.warning("[DEBUG] 7. results empty or None")

        log.info(f"[DEBUG] 8. Done, {len(ocr_results)} results")
        return ocr_results

    except Exception as e:
        log.error(f"[ERROR] Exception in process_ocr: {e}", exc_info=True)
        raise
    finally:
        gc.collect()


# ----------------------------
# API Routes
# ----------------------------
@app.route("/ocr", methods=["POST"])
@limiter.limit(OCR_RATE_LIMIT)
def ocr_endpoint():
    """
    POST /ocr — Main OCR endpoint.

    Input:  {"image_path": "/abs/path/to/image.png",
             "lang": "latin",          # optional, defaults to default_language
             "preprocess": "fast"}     # optional, "fast" (default) or "full"
    Output: {"success": true, "results": [{text, confidence, bbox}, ...],
             "recognition_time": 0.234, "image_path": "...",
             "lang": "latin", "preprocess": "fast"}

    Logic:
        1. Parse JSON body, extract image_path, lang, preprocess
        2. Validate: path exists + extension allowed + lang enabled + mode valid
        3. Call process_ocr(path, lang, mode) and time it
        4. Return results or an appropriate HTTP error code

    Error codes: 400 (no path / bad lang / bad mode), 404 (file missing),
                 415 (bad extension), 500 (OCR blew up).
    """
    data = request.get_json(silent=True) or {}
    image_path = data.get("image_path")

    if not image_path:
        return jsonify({"success": False, "error": "No image_path provided"}), 400
    if not os.path.exists(image_path):
        return jsonify({"success": False, "error": f"Image not found: {image_path}"}), 404
    if not allowed_file(image_path):
        return jsonify({"success": False, "error": "Extension not allowed"}), 415

    lang, mode, err = _validate_lang_mode(data.get("lang"), data.get("preprocess"))
    if err:
        return err

    img_name = os.path.basename(image_path)
    log.info(f"[REQ] OCR requested for: {img_name} (lang={lang}, mode={mode})")

    try:
        t0 = time.time()
        results = process_ocr(image_path, lang, mode)
        dt = time.time() - t0

        resp = {
            "success": True,
            "results": results,
            "recognition_time": round(dt, 3),
            "image_path": image_path,
            "lang": lang,
            "preprocess": mode,
        }
        log.info(f"[OK] OCR {img_name} in {dt:.3f}s, {len(results)} text(s)")
        return jsonify(resp), 200

    except Exception as e:
        log.error(f"[ERR] OCR {img_name}: {e}")
        return jsonify({"success": False, "error": "Internal OCR error"}), 500


@app.route("/status", methods=["GET"])
def status():
    """
    GET /status — Health check.

    Returns whether the OCR engine is loaded and how long the server has been up.
    Used by monitoring, by clients before sending the first real request,
    and by devs who just want to confirm the process is alive.
    """
    return jsonify({
        "status": "ready" if ENGINES else "loading",
        "uptime": round(time.time() - start_time, 2) if start_time else 0,
        "loaded_engines": [f"{lang}:{mode}" for (lang, mode) in ENGINES.keys()],
    }), 200


@app.route("/languages", methods=["GET"])
def languages_endpoint():
    """
    GET /languages — Introspection endpoint.

    Returns the list of enabled languages and the default. Useful for clients
    that want to discover at runtime which (lang, preprocess) combinations
    will be accepted by /ocr.
    """
    return jsonify({
        "enabled": ENABLED_LANGUAGES,
        "default_language": DEFAULT_LANGUAGE,
        "supported": list(SUPPORTED_LANGUAGES),
    }), 200


@app.route("/ocr_strikethrough", methods=["POST"])
@limiter.limit(OCR_RATE_LIMIT)
def ocr_strikethrough_endpoint():
    """
    POST /ocr_strikethrough — OCR + strikethrough detection in a single call.

    Input:  {"image_path": "...", "text": "search_term", "lang": "latin"?}
    Output: {"success": true, "found": true/false, "strikethrough": true/false,
             "text": "...", "confidence": 0.xx, "bbox": [...]}

    Logic:
        1. Run full OCR on the image (via process_ocr) — always 'fast' mode here
        2. Load the same image with cv2.imread for pixel analysis
        3. Iterate through OCR results, looking for the search text (case-insensitive)
        4. When found, crop the region using the bbox corners
        5. Pass that region to detect_horizontal_line() for morphological analysis
        6. Return {found: true, strikethrough: true/false, ...}
        7. If nothing matches, return {found: false, strikethrough: false}

    Why this endpoint exists: some legacy UIs display cancelled/disabled items
    as struck-through text instead of using an explicit state flag. Detecting
    that from the client would have required exposing OpenCV morphology through
    a Java bridge. This endpoint exists because that would have been worse.

    Known limitation: we load the image twice (once for PaddleOCR, once for OpenCV).
    Not optimal. Functional. Nobody has ever complained about the performance.
    """
    data = request.get_json(silent=True) or {}
    image_path = data.get("image_path")
    search_text = (data.get("text") or "").lower()

    if not image_path or not os.path.exists(image_path):
        return jsonify({"success": False, "error": "Image not found"}), 404

    lang, mode, err = _validate_lang_mode(data.get("lang"), "fast")
    if err:
        return err

    try:
        # Step 1: classic OCR pass
        results = process_ocr(image_path, lang, mode)

        # Step 2: load image for OpenCV pixel-level analysis
        img = cv2.imread(image_path)

        # Step 3 & 4: find the text, extract its region
        for item in results:
            if search_text in item["text"].lower():
                bbox = item["bbox"]
                x1, y1 = int(bbox[0][0]), int(bbox[0][1])
                x2, y2 = int(bbox[2][0]), int(bbox[2][1])
                roi = img[y1:y2, x1:x2]

                # Step 5: morphological line detection on the cropped region
                is_strikethrough = detect_horizontal_line(roi)

                return jsonify({
                    "success": True,
                    "found": True,
                    "text": item["text"],
                    "strikethrough": is_strikethrough,
                    "confidence": item["confidence"],
                    "bbox": bbox,
                    "lang": lang,
                })

        # Step 7: text not found
        return jsonify({"success": True, "found": False, "strikethrough": False, "lang": lang})

    except Exception as e:
        log.error(f"[ERR] OCR strikethrough: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


def detect_horizontal_line(roi):
    """
    Detects whether a text region contains a strikethrough line.

    Yes. A strikethrough detector. In an OCR server.
    For a real production use case. We stopped being surprised.

    Logic:
        1. Convert ROI to grayscale if needed
        2. Binarize with Otsu (auto-threshold — no manual calibration per UI)
        3. Apply morphological opening with a horizontal kernel of width w//3
           (one third of the word width, empirically effective)
        4. Extract the vertical center band (between h//3 and 2h//3)
        5. Count white pixels in that band
        6. If more than 30% of width is covered → strikethrough detected

    The 30% threshold was found by iterating on real captures until the
    detection was reliable. It's not in a config file. It's not in a named
    constant. It's in this comment. That's better than nothing.

    Returns: True if a strikethrough line is detected, False otherwise.
    Also returns False if the ROI is too small to analyze (h < 5 or w < 10)
    which usually means something went wrong upstream but that's not our
    problem at this layer.
    """
    if roi is None or roi.size == 0:
        return False

    h, w = roi.shape[:2]
    if h < 5 or w < 10:
        return False

    # Step 1: grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

    # Step 2: binarize (inverted so text becomes white on black)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 3: horizontal kernel — 1/3 of the ROI width, 2 pixels tall
    kernel_width = max(w // 3, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 2))

    # Step 4: morphological opening isolates horizontal line structures
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Step 5: focus on the central band where strikethroughs live
    center_zone = horizontal[h//3: 2*h//3, :]

    # Step 6: count white pixels, compare to 30% of width threshold
    pixel_count = np.sum(center_zone > 0)
    threshold = w * 0.3

    return pixel_count > threshold


# ----------------------------
# CLI / main entry point
# ----------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="paddleocrserver-powered",
        description="PaddleOCR HTTP server with multi-language support.",
    )
    parser.add_argument("--host", default=os.getenv("OCR_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("OCR_PORT", "5000")))
    parser.add_argument(
        "--languages-config",
        default=None,
        help="Path to a custom languages.yaml (overrides OCR_LANGUAGES_CONFIG).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Boot sequence:
        1. Parse CLI flags
        2. Bail out if the chosen port is already taken (we've been there)
        3. Resolve languages.yaml (CLI > env var > packaged default)
        4. Eager-load every enabled language in 'fast' mode so the first
           request never pays a cold-start cost. 'full' stays lazy.
        5. Hand off to Waitress, threads=1 (we tried 4. we don't talk about it.)
    """
    global ENABLED_LANGUAGES, DEFAULT_LANGUAGE, start_time

    args = _parse_args(argv)

    if is_port_in_use(args.host, args.port):
        print(f"[WARN] Port {args.host}:{args.port} already in use, server probably already running")
        return 0

    cfg_path = resolve_languages_config(args.languages_config)
    log.info(f"[CFG] Loading languages config from {cfg_path}")
    ENABLED_LANGUAGES, DEFAULT_LANGUAGE = load_languages_config(cfg_path)
    log.info(f"[CFG] Enabled: {ENABLED_LANGUAGES} (default={DEFAULT_LANGUAGE})")

    log.info("[INFO] Waking up the neural beast...")
    log.info("[INFO] Baidu spent 4 years. We did pip install. That's progress.")

    start_time = time.time()
    for lang in ENABLED_LANGUAGES:
        get_engine(lang, "fast")

    # PaddleOCR loads asynchronously — give it a moment to finish complaining
    time.sleep(1)
    elapsed = time.time() - start_time

    log.info("=" * 55)
    log.info(f"[OK] PaddleOCR ready in {elapsed:.2f}s -- faster than most legacy OSes boot.")
    log.info(f"[CFG] Engines loaded (fast): {len(ENGINES)}  /  Languages: {ENABLED_LANGUAGES}")
    log.info("[CFG] Waitress threads: 1  (we tried 4. we don't talk about it.)")
    log.info(f"[NET] Listening on http://{args.host}:{args.port} — clients can send images now.")
    log.info("=" * 55)

    serve(app, host=args.host, port=args.port, threads=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
