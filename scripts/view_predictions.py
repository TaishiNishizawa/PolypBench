"""Browse a REAL-Colon video frame by frame with model predictions and GT boxes overlaid.

    python scripts/view_predictions.py --dataset realcolon_val --video 004-012 \
        --models emcad-b2@sunseg-prime sali@sunseg-causal

    In another iTerm: ssh -N -L 8765:localhost:8765 tn23794@ece-a55005.austin.utexas.edu


The page shows, per model, the frame with the >= threshold region tinted and
outlined (green) and the GT boxes (white), plus a timeline of predicted area per
frame to click on. Logic lives in polypbench/viz/frame_viewer.py.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2

from polypbench.viz.frame_viewer import PAGE, FrameViewer


def make_handler(viewer):
    page = PAGE.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body, content_type):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/":
                    self._send(page, "text/html; charset=utf-8")
                elif url.path == "/meta":
                    self._send(json.dumps(viewer.meta()).encode(), "application/json")
                elif url.path == "/stats":
                    self._send(json.dumps(viewer.stats(int(q["i"]), float(q["t"]))).encode(), "application/json")
                elif url.path == "/frame":
                    body = viewer.render(q["model"], int(q["i"]), float(q["t"]), q.get("gt", "1") == "1")
                    self._send(body, "image/jpeg")
                else:
                    self.send_error(404)
            except (KeyError, ValueError, FileNotFoundError) as e:
                self.send_error(400, str(e))

        def log_message(self, *args):   # quiet: one line per image request is noise
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="e.g. realcolon_val")
    parser.add_argument("--video", required=True, help="e.g. 004-012")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--postproc", default="none")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    cv2.setNumThreads(2)

    print(f"Loading {args.dataset}/{args.video} and the timeline for {len(args.models)} model(s)...", flush=True)
    viewer = FrameViewer(args.dataset, args.video, args.models, args.postproc)
    if viewer.keep_all:
        print(f"All predictions held in memory ({viewer.memory_gb:.1f} GB): every frame loads instantly.")
    else:
        print(f"Predictions would need {viewer.memory_gb:.0f} GB, so only recent chunks are kept: "
              f"jumping far takes ~10 s per model.")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(viewer))
    print(f"Ready: http://localhost:{args.port}  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
