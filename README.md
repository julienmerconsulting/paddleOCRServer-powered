# paddleocrserver-powered

[![PyPI version](https://img.shields.io/pypi/v/paddleocrserver-powered.svg)](https://pypi.org/project/paddleocrserver-powered/)
[![Python versions](https://img.shields.io/pypi/pyversions/paddleocrserver-powered.svg)](https://pypi.org/project/paddleocrserver-powered/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A self-contained, multi-language **PaddleOCR HTTP server** packaged as a single
PyPI install. Flask + Waitress, configurable per-language YAML pipelines, and
two preprocessing modes (`fast` for screens/UI/printed docs, `full` for
photographed/curved/rotated documents).

```
Any client (Java, Python, RF, curl)  →  HTTP POST  →  Flask  →  PaddleOCR  →  JSON
```

---

## Install

```bash
pip install paddleocrserver-powered
```

> The first request for a given (language, mode) pair downloads the
> corresponding PaddleOCR model weights into the local PaddleX cache.

## Quickstart

```bash
paddleocrserver-powered
# → serves http://127.0.0.1:5000  (latin language, fast mode)
```

Run a quick OCR:

```bash
curl -X POST http://127.0.0.1:5000/ocr \
  -H 'Content-Type: application/json' \
  -d '{"image_path": "/abs/path/to/image.png"}'
```

You can also run it as a module:

```bash
python -m paddleocrserver_powered --port 5000
```

---

## OCR modes

| Mode   | Pipeline flags enabled                                              | Latency (indicative) | Use cases                                                  |
| ------ | ------------------------------------------------------------------- | -------------------- | ---------------------------------------------------------- |
| `fast` | text detection + recognition only                                   | ~150–400 ms / image  | UI screenshots, screen captures, clean printed documents   |
| `full` | + doc orientation classify + textline orientation + doc unwarping   | ~600–1500 ms / image | Photographed docs, curved/rotated pages, low-quality scans |

`fast` is the default. Pass `"preprocess": "full"` in the request body to
enable the full pipeline for a single call.

---

## Languages

Built-in language packs (PP-OCRv5 mobile recognition models):

| Code          | Recognition model                       |
| ------------- | --------------------------------------- |
| `latin`       | `latin_PP-OCRv5_mobile_rec` (default)   |
| `japan`       | `japan_PP-OCRv5_mobile_rec`             |
| `korean`      | `korean_PP-OCRv5_mobile_rec`            |
| `chinese_cht` | `chinese_cht_PP-OCRv5_mobile_rec`       |
| `cyrillic`    | `cyrillic_PP-OCRv5_mobile_rec`          |
| `arabic`      | `arabic_PP-OCRv5_mobile_rec`            |
| `devanagari`  | `devanagari_PP-OCRv5_mobile_rec`        |

By default **only `latin` is enabled**. Activate other languages by editing
`languages.yaml`:

```yaml
enabled:
  latin: true
  japan: true
  korean: false
  chinese_cht: false
  cyrillic: false
  arabic: false
  devanagari: false
default_language: latin
```

Then point the server at it:

```bash
paddleocrserver-powered --languages-config /path/to/languages.yaml
# or
OCR_LANGUAGES_CONFIG=/path/to/languages.yaml paddleocrserver-powered
```

All `enabled: true` languages are **eager-loaded** at startup in `fast` mode,
so the first request never pays a cold-start cost. The corresponding `full`
engine is created lazily the first time someone requests `preprocess: "full"`.

### Adding languages

To add a custom language pack:

1. Drop two YAML files next to the packaged ones (or beside your custom
   `languages.yaml`): `ocr_config_<code>.yaml` and `ocr_config_<code>_full.yaml`.
2. Set the appropriate `model_name` under `SubModules.TextRecognition` (e.g.
   `myscript_PP-OCRv5_mobile_rec`).
3. Add `<code>: true` to your `languages.yaml`.

The shipped YAMLs are a safe template — copy `ocr_config_latin.yaml`, change
the `model_name`, you're done.

---

## API endpoints

### `POST /ocr`

Run OCR on an image.

```bash
curl -X POST http://127.0.0.1:5000/ocr \
  -H 'Content-Type: application/json' \
  -d '{
    "image_path": "/abs/path/to/image.png",
    "lang": "latin",
    "preprocess": "fast"
  }'
```

| Field         | Type   | Required | Default            | Notes                                |
| ------------- | ------ | -------- | ------------------ | ------------------------------------ |
| `image_path`  | string | yes      | —                  | Absolute path on the server side     |
| `lang`        | string | no       | `default_language` | Must be in the `enabled` list        |
| `preprocess`  | string | no       | `"fast"`           | `"fast"` or `"full"`                 |

If `lang` is not enabled the server returns **400** with the list of
currently-enabled languages.

### `POST /ocr_strikethrough`

OCR + strikethrough detection in one call. Always uses `fast` preprocessing.

```bash
curl -X POST http://127.0.0.1:5000/ocr_strikethrough \
  -H 'Content-Type: application/json' \
  -d '{"image_path": "/abs/path/img.png", "text": "cancelled", "lang": "latin"}'
```

### `GET /status`

```bash
curl http://127.0.0.1:5000/status
```

Returns `{status, uptime, loaded_engines}`.

### `GET /languages`

```bash
curl http://127.0.0.1:5000/languages
```

Returns `{enabled, default_language, supported}`.

---

## Configuration

| Env var                     | Default          | Purpose                                          |
| --------------------------- | ---------------- | ------------------------------------------------ |
| `OCR_HOST`                  | `127.0.0.1`      | Bind address                                     |
| `OCR_PORT`                  | `5000`           | Bind port                                        |
| `OCR_LANGUAGES_CONFIG`      | (packaged)       | Path to a custom `languages.yaml`                |
| `OCR_RATE_LIMIT_DEFAULT`    | `100 per minute` | Global rate limit (per IP)                       |
| `OCR_RATE_LIMIT_OCR`        | `30 per minute`  | Per-IP limit on `/ocr` and `/ocr_strikethrough`  |
| `OCR_RATE_LIMIT_STORAGE`    | `memory://`      | flask-limiter storage URI (e.g. Redis)           |

CLI flags (override env vars): `--host`, `--port`, `--languages-config`.

---

## Advanced: multi-port deployment

PaddleOCR is **not** thread-safe — a single process runs with `threads=1`. To
serve multiple languages efficiently, run **one process per language on its
own port** and route from the client side.

`/etc/paddleocr/latin.yaml`:

```yaml
enabled: { latin: true }
default_language: latin
```

`/etc/paddleocr/japan.yaml`:

```yaml
enabled: { japan: true }
default_language: japan
```

Launch:

```bash
OCR_PORT=5000 paddleocrserver-powered --languages-config /etc/paddleocr/latin.yaml &
OCR_PORT=5001 paddleocrserver-powered --languages-config /etc/paddleocr/japan.yaml &
```

Each instance keeps only its own model in RAM and answers only on its port.

---

## Programmatic usage

```python
from paddleocrserver_powered import spawn_server

proc = spawn_server(port=5050, languages_config="/etc/paddleocr/latin.yaml")
try:
    # ... do work, hit http://127.0.0.1:5050/ocr ...
    pass
finally:
    proc.terminate()
    proc.wait(timeout=10)
```

`spawn_server` returns a `subprocess.Popen` running `python -m
paddleocrserver_powered` with the provided arguments.

---

## License

MIT — see [LICENSE](LICENSE).
