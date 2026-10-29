"""
Local shop website for Gate M: serves the Gate W product pages over real HTTP.

    GET /<domain>/p<page>         product page (current phase)
    GET /_admin/phase/<0|1>       switch phase (1 = after the redesign)

Binds to 127.0.0.1 only.
"""

import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from run_gate_w import products, render, sites  # noqa: E402


class ShopSite:
    def __init__(self, port: int = 0):
        self.phase = 0
        self._sites = {s["domain"]: s for s in sites()}
        self._products = {s["domain"]: products(s) for s in sites()}
        site = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output quiet
                pass

            def do_GET(self):
                m = re.match(r"^/_admin/phase/([01])$", self.path)
                if m:
                    site.phase = int(m.group(1))
                    return self._send(200, "ok", "text/plain")
                m = re.match(r"^/([a-z0-9.]+)/p(\d+)$", self.path)
                if not m or m.group(1) not in site._sites:
                    return self._send(404, "not found", "text/plain")
                domain, page = m.group(1), int(m.group(2))
                prods = site._products[domain]
                if page >= len(prods):
                    return self._send(404, "not found", "text/plain")
                html = render(site._sites[domain], prods[page], site.phase == 1)
                return self._send(200, html, "text/html")

            def _send(self, code, body, ctype):
                data = body.encode()
                self.send_response(code)
                self.send_header("Content-Type", f"{ctype}; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def url(self, domain: str, page: int) -> str:
        return f"{self.base_url}/{domain}/p{page}"

    def start(self) -> "ShopSite":
        self.thread.start()
        return self

    def set_phase(self, phase: int) -> None:
        self.phase = phase

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
