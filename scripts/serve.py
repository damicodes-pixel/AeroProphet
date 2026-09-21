import http.server, socketserver, webbrowser
from pathlib import Path

PORT = 8000
web_dir = str(Path(__file__).parent / "web")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=web_dir, **kwargs)


socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("", PORT), Handler) as server:
    print(f"AeroProphet sim running at http://localhost:{PORT}  (Ctrl+C to stop)")
    webbrowser.open(f"http://localhost:{PORT}")
    server.serve_forever()