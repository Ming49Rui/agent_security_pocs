# -*- coding: utf-8 -*-
"""
2609.01222 复现用的轻量验证端点（纯标准库，零依赖）。

双模式：
  MOCK=1          —— 不请求真实模型，返回预设回复。用于“确定性源验证”：
                     验证某个上下文源的内容是否真的以某个角色进入请求。
  UPSTREAM=...    —— 把请求转发到真实端点（并记录），用于端到端攻击验证。

两种模式都会把收到的请求体(const 的 messages 数组)追加记录到 EVIDENCE_DIR。
用法：
  python endpoint_logger.py --port 18100
环境变量：
  EVIDENCE_DIR  证据目录（默认 ./evidence）
  MOCK          1 表示 mock 模式
  REPLY         mock 回复文本（默认 "Understood."）
  UPSTREAM      上游 https://model.shouxu.tech/v1/chat/completions
  UPSTREAM_KEY  上游 Bearer key（缺省则透传请求自带的 Authorization）
"""
import json
import os
import socketserver
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

EVIDENCE_DIR = os.environ.get("EVIDENCE_DIR", "./evidence")
MOCK = os.environ.get("MOCK") == "1"
REPLY = os.environ.get("REPLY", "Understood.")
UPSTREAM = os.environ.get("UPSTREAM", "")
UPSTREAM_KEY = os.environ.get("UPSTREAM_KEY", "")

os.makedirs(EVIDENCE_DIR, exist_ok=True)
LOGLOCK = threading.Lock()
SEQ = [0]


def log_request(body: bytes, response: dict = None, status: int = 200):
    with LOGLOCK:
        SEQ[0] += 1
        path = os.path.join(EVIDENCE_DIR, f"req_{SEQ[0]:04d}.jsonl")
        rec = {"seq": SEQ[0], "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "mock": bool(MOCK), "status": status}
        if body:
            try:
                rec["request"] = json.loads(body)
            except Exception:
                rec["request_raw"] = body.decode("utf-8", "replace")
        if response is not None:
            rec["response"] = response
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, indent=1))


def sse_event(obj):
    return f"event: {obj.get('type', '')}\ndata: {json.dumps(obj)}\n\n".encode()


def make_responses_reply(text: str, model: str = "deepseek-v4-flash-0731", stream: bool = False):
    resp_id = "resp_mock_cpe_%04d" % SEQ[0]
    base = {
        "id": resp_id,
        "object": "response",
        "created_at": int(time.time()),
        "model": model,
        "status": "completed",
        "background": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "store": True,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}, "gather_evidence": True},
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": None,
        "user": None,
        "metadata": {},
    }
    msg_id = "msg_mock_cpe_%04d" % SEQ[0]
    output_item = {
        "id": msg_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    if not stream:
        return {**base, "output": [output_item]}
    # 流式：返回事件序列
    full = {**base, "output": [output_item]}
    events = [
        {"type": "response.created", "response": full},
        {"type": "response.in_progress", "sequence_number": 0, "response": full},
        {"type": "response.output_item.added",
         "output_index": 0, "item": output_item, "sequence_number": 1},
        {"type": "response.content_part.added",
         "item_id": msg_id, "output_index": 0, "content_index": 0,
         "part": output_item["content"][0], "sequence_number": 2},
        {"type": "response.output_text.delta",
         "item_id": msg_id, "output_index": 0, "content_index": 0, "delta": text,
         "sequence_number": 3},
        {"type": "response.output_text.done",
         "item_id": msg_id, "output_index": 0, "content_index": 0, "text": text,
         "sequence_number": 4},
        {"type": "response.content_part.done",
         "item_id": msg_id, "output_index": 0, "content_index": 0,
         "part": output_item["content"][0], "sequence_number": 5},
        {"type": "response.output_item.done",
         "output_index": 0, "item": output_item, "sequence_number": 6},
        {"type": "response.completed", "response": full, "sequence_number": 7},
    ]
    return events


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        return self.rfile.read(length) if length else b""

    def _send_json(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_sse(self, chunks, done=True):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for c in chunks:
            self.wfile.write(c)
            self.wfile.flush()
        if done:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        self.close_connection = True

    def do_POST(self):
        body = self._read_body()
        if MOCK:
            self._handle_mock(body)
        else:
            self._handle_proxy(body)

    def _handle_mock(self, body):
        stream = False
        model = "deepseek-v4-flash-0731"
        if body:
            try:
                req = json.loads(body)
                stream = bool(req.get("stream"))
                model = req.get("model", model)
            except Exception:
                pass
        log_request(body, status=200)
        text = REPLY
        if stream:
            chunks = [sse_event(ev) for ev in make_responses_reply(text, model, stream=True)]
            self._send_sse(chunks)
        else:
            self._send_json(make_responses_reply(text, model, stream=False))

    def _handle_proxy(self, body):
        # 解析请求信息用于记录
        resp_status = 502
        detail = None
        try:
            req = json.loads(body) if body else {}
            stream = bool(req.get("stream"))
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + (UPSTREAM_KEY or ""),
            }
            if not headers["Authorization"].startswith("Bearer "):
                # 透传请求自带的鉴权头
                incoming = self.headers.get("Authorization", "")
                if incoming:
                    headers["Authorization"] = incoming
            url = UPSTREAM
            rq = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(rq, timeout=300) as up:
                resp_status = up.status
                ctype = up.headers.get("Content-Type", "")
                if stream:
                    self.send_response(up.status)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    buf = bytearray()
                    while True:
                        chunk = up.read(65536)
                        if not chunk:
                            break
                        buf += chunk
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    self.close_connection = True
                    detail = {"streamed_bytes": len(buf)}
                else:
                    data = up.read()
                    detail = json.loads(data) if data else None
                    self._send_json(detail, status=up.status)
        except Exception as e:
            detail = {"proxy_error": str(e)}
            self._send_json({"error": {"message": str(e)}}, status=resp_status)
        log_request(body, response=detail, status=resp_status)

    def log_message(self, fmt, *args):
        pass  # 静默


class QuietServer(ThreadingHTTPServer):
    """绕过 Windows 上 socket.getfqdn 的 UnicodeDecodeError"""

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = host
        self.server_port = port


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18100)
    args = ap.parse_args()
    print(f"[endpoint_logger] EVIDENCE_DIR={EVIDENCE_DIR} MOCK={MOCK} "
          f"UPSTREAM={'set' if UPSTREAM else 'unset'}")
    QuietServer(("127.0.0.1", args.port), Handler).serve_forever()