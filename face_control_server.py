# coding=utf-8
"""Local-only, newline-delimited JSON TCP transport. No camera or database access."""
import json
import socket

HOST = "127.0.0.1"
PORT = 5055
MAX_REQUEST_BYTES = 4096


def run_server(handler):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(5)
        server.settimeout(0.5)  # Allow Ctrl+C while idle on Windows.
        print(f"TCP server ready: {HOST}:{PORT}", flush=True)
        print("Waiting for Unity commands...", flush=True)
        while True:
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            with conn:
                conn.settimeout(5)
                try:
                    # One JSON request per connection, terminated by newline.
                    line = bytearray()
                    while len(line) <= MAX_REQUEST_BYTES:
                        part = conn.recv(1)
                        if not part or part == b"\n":
                            break
                        line.extend(part)
                    if not line or len(line) > MAX_REQUEST_BYTES:
                        response = {"ok": False, "error": "Invalid or oversized request"}
                    else:
                        response = handler(json.loads(line.decode("utf-8")))
                except (UnicodeError, ValueError, socket.timeout, OSError):
                    response = {"ok": False, "error": "Invalid request"}
                try:
                    conn.sendall((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
                except OSError:
                    pass  # Client disconnected.


if __name__ == "__main__":
    print("Run via face_login_service.py --mode server")
