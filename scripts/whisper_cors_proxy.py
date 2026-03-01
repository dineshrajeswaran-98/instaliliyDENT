#!/usr/bin/env python3
"""WhisperKit CORS proxy for DefectBot.

WhisperKit's serve command doesn't emit Access-Control-Allow-Origin headers,
so browsers block responses when the app is opened as file://.
This proxy sits on port 2022, forwards POST requests to WhisperKit on port 2023,
and injects the required CORS headers so the browser can read the transcript.

Usage (called by setup_mac.sh):
    python3 scripts/whisper_cors_proxy.py
"""

import http.server
import urllib.request
import urllib.error
import sys

WHISPER_BACKEND = 'http://127.0.0.1:2023'
PROXY_PORT = 2022


class WhisperProxy(http.server.BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length else b''

        # Forward all headers except host (let urllib set it correctly)
        forward_headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in ('host', 'content-length')
        }

        req = urllib.request.Request(
            WHISPER_BACKEND + self.path,
            data=body,
            headers=forward_headers,
            method='POST',
        )

        try:
            with urllib.request.urlopen(req) as resp:
                data = resp.read()
                self.send_response(resp.status)
                self._cors()
                self.send_header('Content-Type',
                                 resp.headers.get('Content-Type', 'application/json'))
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as exc:
            data = exc.read()
            self.send_response(exc.code)
            self._cors()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(data)

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def log_message(self, fmt, *args):
        pass  # silence request logs


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PROXY_PORT
    server = http.server.HTTPServer(('127.0.0.1', port), WhisperProxy)
    print(f'[DefectBot] WhisperKit CORS proxy → port {port} → {WHISPER_BACKEND}',
          flush=True)
    server.serve_forever()
