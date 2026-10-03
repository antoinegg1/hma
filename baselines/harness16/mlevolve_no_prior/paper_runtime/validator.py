"""Loopback submission format service that only reads the public sample."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

from .artifacts import validate_csv


def create_app(public: Path, slug: str) -> Flask:
    """Preserve the upstream response schema without invoking hidden grading."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024**3

    @app.get("/health")
    def health() -> Any:
        return jsonify({"status": "running", "feedback": "public_schema_only"})

    @app.post("/validate")
    def validate() -> Any:
        if request.headers.get("exp-id") != slug or "file" not in request.files:
            return jsonify(
                {"error": "invalid request", "details": "task or file mismatch"}
            ), 400
        with tempfile.TemporaryDirectory(prefix="paper-format-") as directory:
            path = Path(directory) / "submission.csv"
            request.files["file"].save(path)
            valid, message = validate_csv(path, public)
        return jsonify({"is_valid": valid, "result": message})

    return app


def main() -> None:
    """Serve concurrent requests with distinct temporary files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    create_app(args.public, args.slug).run(
        host="127.0.0.1", port=args.port, threaded=True
    )


if __name__ == "__main__":
    main()
