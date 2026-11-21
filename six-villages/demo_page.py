#!/usr/bin/env python3
"""
Six Villages, One City: demo page

The same villages as poc.py, but in the browser, so a visitor can type and
see which facts each answer used. Nothing new to install: Python + Ollama.

  python3 demo_page.py                 then open http://localhost:8000
  python3 demo_page.py --model gemma3:4b
  python3 demo_page.py --port 8080

Every answer is still saved in logs/truth_check.csv, exactly like the terminal.
If the page ever misbehaves during a demo, the terminal is the fallback:
  python3 poc.py gestel --with strijp
"""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import poc   # all the real work (prompts, language, citations, log) lives in poc.py

PAGE = Path(__file__).resolve().parent / "web" / "index.html"


def build_state(args):
    """Loads every village once and fetches its live data once, when the server starts."""
    villages = {name: poc.load_village(name) for name in poc.list_villages()}
    live = {} if args.no_live else {name: poc.live_facts_for(v) for name, v in villages.items()}
    return {"villages": villages, "live": live, "model": args.model,
            "no_think": args.no_think, "lock": threading.Lock()}


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):          # keep the Terminal quiet; answers are in the CSV
            pass

        def send_json(self, data, code=200):
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = PAGE.read_bytes()    # read on every load, so edits show after a refresh
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/villages":
                # What the page needs to draw its menus: names, fact counts, live facts.
                self.send_json({"model": state["model"], "villages": [
                    {"name": n, "label": v["label"], "facts": len(v["facts"]),
                     "live": [f["fact"] for f in state["live"].get(n, [])]}
                    for n, v in state["villages"].items()]})
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path not in ("/ask", "/match"):
                return self.send_error(404)
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
            previous = req.get("previous_lang") if req.get("previous_lang") in ("en", "nl") else "en"

            if self.path == "/match":
                profile = (req.get("profile") or "").strip()
                if not profile:
                    return self.send_json({"error": "Describe yourself or click a Match preset first."}, 400)
                lang = req.get("lang") if req.get("lang") in ("en", "nl") else \
                    poc.detect_language(profile, fallback=previous)
                try:
                    with state["lock"]:
                        result = poc.run_match(state["model"], state["no_think"],
                                               list(state["villages"].values()),
                                               state["live"], profile, lang)
                except SystemExit as e:
                    return self.send_json({"error": str(e)}, 502)
                return self.send_json(result)

            question = (req.get("question") or "").strip()
            home = state["villages"].get(req.get("village"))
            guest = state["villages"].get(req.get("guest")) if req.get("guest") else None
            if not question or not home or (guest and guest["name"] == home["name"]):
                return self.send_json({"error": "Pick a village and type a question."}, 400)

            # Language: forced by the page, or picked from the question (ties keep the last one).
            lang = req.get("lang") if req.get("lang") in ("en", "nl") else \
                poc.detect_language(question, fallback=previous)
            # Only the last three exchanges, like the terminal.
            history = [m for m in req.get("history", [])
                       if m.get("role") in ("user", "assistant")][-6:]
            try:
                # One question at a time: a laptop runs one model, two at once only slows both.
                with state["lock"]:
                    result = poc.run_turn(state["model"], state["no_think"], home,
                                          state["live"].get(home["name"], []), guest,
                                          state["live"].get(guest["name"], []) if guest else [],
                                          history, question, lang)
            except SystemExit as e:          # poc.ask_model exits with a readable message
                return self.send_json({"error": str(e)}, 502)
            self.send_json(result)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Six Villages demo page")
    parser.add_argument("--model", default=poc.DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--no-live", action="store_true")
    args = parser.parse_args()

    state = build_state(args)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    names = ", ".join(v["label"] for v in state["villages"].values())
    print(f"Villages: {names}. Model: {args.model}.")
    print(f"Open http://localhost:{args.port} in your browser. Ctrl+C stops it.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        poc.unload_model(args.model)   # free the 8 GB as soon as the demo stops


if __name__ == "__main__":
    main()
