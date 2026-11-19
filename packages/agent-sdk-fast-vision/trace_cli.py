"""Read back what the fast vision loop did, one model call at a time.

    python packages/agent-sdk-fast-vision/trace_cli.py                 # last 10 calls in $ANO_RUN_DIR (default runs/fast-vision)
    python packages/agent-sdk-fast-vision/trace_cli.py --last 30
    python packages/agent-sdk-fast-vision/trace_cli.py --call 17 --full   # one call, prompt included
    python packages/agent-sdk-fast-vision/trace_cli.py --follow           # print calls as they finish

For each call it shows:
  1. the 6 frames aggregated into the image, with capture time, and when the image was sent
  2. the exact API input  (trace/<call>/request.json; the image is trace/<call>/sheet.png)
  3. the exact API output (trace/<call>/response.json; the model's text is shown)
  4. what was applied to the game, or why nothing was
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def load(run: Path) -> List[Dict[str, Any]]:
    path = run / "trace.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text("utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return sorted(rows, key=lambda r: r.get("call", 0))


def describe_request(run: Path, row: Dict[str, Any], full: bool) -> List[str]:
    path = run / row.get("dir", "") / "request.json"
    if not path.exists():
        return [f"     (request.json pruned or not written: {path})"]
    body = json.loads(path.read_bytes())
    lines = [f"     file  {path}  ({path.stat().st_size} bytes)",
             f"     model={body.get('model')} max_tokens={body.get('max_tokens')} "
             f"temperature={body.get('temperature')}"]
    for message in body.get("messages", []):
        content = message.get("content")
        if isinstance(content, str):
            if full:
                lines.append(f"     [{message['role']}]")
                lines += ["       " + part for part in content.splitlines()]
            else:
                digest = hashlib.sha1(content.encode()).hexdigest()[:8]
                lines.append(f"     [{message['role']}] {len(content)} chars (sha {digest}; --full to print)")
            continue
        for part in content:
            if part.get("type") == "text":
                lines.append(f"     [{message['role']}] text: {part['text']}")
            elif part.get("type") == "image_url":
                url = part["image_url"]["url"]
                raw = base64.b64decode(url.split(",", 1)[1]) if "," in url else b""
                size = ""
                if raw[:8] == b"\x89PNG\r\n\x1a\n" and len(raw) > 24:
                    size = f"{int.from_bytes(raw[16:20], 'big')}x{int.from_bytes(raw[20:24], 'big')} "
                lines.append(f"     [{message['role']}] image: PNG {size}{len(raw)} bytes, "
                             f"detail={part['image_url'].get('detail')} -> {run / row.get('dir', '')}/sheet.png")
    return lines


def show(run: Path, row: Dict[str, Any], full: bool) -> str:
    out = [f"=== call {row.get('call')}  {row.get('dir', '')}"]
    frames = row.get("frames") or []
    out.append(f"  1. image = {len(frames)} frames aggregated, sent {row.get('sent', '?')}")
    for frame in frames:
        after = f"  then pressed {frame['key_after']}" if frame.get("key_after") else ""
        out.append(f"       frame #{frame['seq']:<6} captured {frame['captured']}  "
                   f"(age {frame['age_ms']} ms when sent){after}")
    out.append("  2. API input")
    out += describe_request(run, row, full)
    out.append(f"  3. API output  received {row.get('received', '?')}  "
               f"({row.get('model_ms')} ms, HTTP {row.get('http')})")
    if row.get("output"):
        out.append(f"       model text: {row['output']}")
    if row.get("error"):
        out.append(f"       error: {row['error']}")
    out.append(f"       file {run / row.get('dir', '')}/response.json")
    outcome = row.get("outcome", "?")
    if outcome == "applied":
        out.append(f"  4. applied {row.get('applied')}: {', '.join(row.get('sent_to_game') or [])}"
                   f"  (answer waited {row.get('wait_ms')} ms)")
    else:
        out.append(f"  4. nothing sent to the game: {outcome}")
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", default=os.environ.get("ANO_RUN_DIR", os.path.join(os.getcwd(), "runs", "fast-vision")))
    parser.add_argument("--last", type=int, default=10)
    parser.add_argument("--call", type=int)
    parser.add_argument("--full", action="store_true", help="print the system prompt too")
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args(argv)
    run = Path(args.run)
    rows = load(run)
    if args.call is not None:
        rows = [r for r in rows if r.get("call") == args.call]
        if not rows:
            print(f"call {args.call} not in {run / 'trace.jsonl'}", file=sys.stderr)
            return 1
    elif not args.follow:
        rows = rows[-args.last:]
    if not rows and not args.follow:
        print(f"no calls traced yet in {run / 'trace.jsonl'}", file=sys.stderr)
        return 1
    for row in rows:
        print(show(run, row, args.full), end="\n\n")
    if args.follow:
        seen = {r.get("call") for r in load(run)}
        while True:
            time.sleep(0.5)
            for row in load(run):
                if row.get("call") not in seen:
                    seen.add(row.get("call"))
                    print(show(run, row, args.full), end="\n\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
