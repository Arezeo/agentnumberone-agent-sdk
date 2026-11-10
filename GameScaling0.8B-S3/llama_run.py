#!/usr/bin/env python3
"""Run the reference WebRTC runner with the llama.cpp GameScaling policy.

Start a llama.cpp server with a GameScaling GGUF and its mmproj first; this
script neither downloads weights nor starts that server. It keeps the pairing
token in the kit's private `connection.txt`, so the paired URL stays the same
across restarts, and prints the file's location, never the token.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import urllib.request
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from connection import pairing_token  # noqa: E402
from model_catalog import data_dir  # noqa: E402

GAMES = ("auto", "general", "voxelcraft", "pixel-duel")


def control_url(value):
    parsed = urlsplit(value)
    if parsed.scheme not in ("ws", "wss") or not parsed.hostname or parsed.username or parsed.password \
            or parsed.query or parsed.fragment or parsed.path != "/control":
        raise argparse.ArgumentTypeError("Use ws(s)://HOST[:PORT]/control, without credentials or token")
    return value


def server_ready(base):
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/health", timeout=5) as response:
            return json.load(response).get("status") == "ok"
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-url", required=True, type=control_url,
                        help="Endpoint the game page connects to, without a token. A hosted HTTPS page needs trusted wss://")
    parser.add_argument("--game", choices=GAMES, default="auto", help="Built-in system prompt")
    parser.add_argument("--prompt-file", type=Path,
                        help="Your own system prompt for another game; overrides --game")
    parser.add_argument("--server", default="http://127.0.0.1:8901", help="llama.cpp server base URL")
    parser.add_argument("--origin", default="https://agentnumberone.io")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    if args.prompt_file and not (args.prompt_file.is_file() and args.prompt_file.read_text().strip()):
        raise SystemExit("--prompt-file must be a readable, non-empty text file: " + str(args.prompt_file))
    if not server_ready(args.server):
        raise SystemExit("No llama.cpp server answered at " + args.server + "/health. Start it first.")
    connection = data_dir() / "connection.txt"
    token = pairing_token(connection, args.public_url)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "ARENA_CONTROL_TOKEN": token,
           "ANO_LLAMA_URL": args.server.rstrip("/") + "/v1/chat/completions", "ANO_LLAMA_GAME": args.game}
    if args.prompt_file:
        env["ANO_LLAMA_PROMPT_FILE"] = str(args.prompt_file.resolve())
    print(json.dumps({"status": "starting", "game": "custom" if env.get("ANO_LLAMA_PROMPT_FILE") else args.game,
                      "connection_file": str(connection),
                      "live_view": "http://127.0.0.1:%d/live" % args.port}), flush=True)
    # The runner prints its own pairing line with the token; keep it out of this terminal and any log.
    process = subprocess.Popen(
        [sys.executable, str(HERE.parent / "scripts/remote-control/agent.py"), "--origin", args.origin,
         "--port", str(args.port), "--policy", str(HERE / "llama_policy.py"), "--live-view",
         "--decision-interval-ms", "0", "--inference-timeout-ms", "120000"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)     # stopping this launcher must stop the runner too
    try:
        for line in process.stdout:
            if "token=" not in line:
                print(line, end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
    return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
