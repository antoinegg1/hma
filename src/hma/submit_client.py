"Blind submit helper, authenticated only for this session's evaluator turn."

import argparse
import http.client
import json
from pathlib import Path

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["submit", "candidate", "validate", "status", "scores"]
    )
    parser.add_argument("value", nargs="?")
    args = parser.parse_args()
    route = json.loads(Path("/run/hma/route.json").read_text())
    connection = http.client.HTTPConnection(route["host"], route["port"], timeout=3600)
    headers = {"X-Turn-Token": route["token"]}
    if args.command in {"submit", "candidate", "validate"}:
        path = Path(args.value or "")
        if path.is_symlink() or not path.is_file():
            parser.error("artifact must be a regular file")
        headers.update(
            {"Content-Length": str(path.stat().st_size), "Content-Type": "text/csv"}
        )
        with path.open("rb") as artifact:
            connection.request(
                "POST",
                "/validate" if args.command == "validate" else "/submit",
                body=artifact,
                headers=headers,
            )
    else:
        connection.request(
            "GET",
            "/scores" if args.command == "scores" else f"/submissions/{args.value}",
            headers=headers,
        )
    response = connection.getresponse()
    content = response.read(4 * 1024 * 1024 + 1)
    connection.close()
    if len(content) > 4 * 1024 * 1024:
        raise ValueError("evaluator response too large")
    print(json.dumps(json.loads(content), sort_keys=True))
    return 0 if 200 <= response.status < 300 else 1

if __name__ == "__main__":
    raise SystemExit(main())
