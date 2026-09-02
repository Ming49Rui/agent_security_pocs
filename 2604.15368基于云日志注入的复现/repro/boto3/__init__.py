"""Shadow replacement for the AWS SDK `boto3`: serves the LogJack payloads
from benchmark/logjack-30.json with the same response shapes as real boto3,
without provisioning anything or requiring AWS credentials.

Data mapping mirrors the original benchmark's provision.py + build_prompt:
  - cloudwatch payloads        -> log group  /logjack/individual/{pid}
  - lambda (rce-/ctrl-)        -> log group  /logjack/individual/{pid}
  - lambda (attack)            -> log group  /logjack/benchmark-lambda
  - ssm_error_report/cloudtrail-> SSM param  /logjack/benchmark/{pid}
  - cicd                       -> S3 object  s3://logjack-benchmark-730335305775/cicd-logs/{pid}.log
"""

import io
import json
import os
import time

_BUCKET = "logjack-benchmark-730335305775"


class ClientError(Exception):
    """Mimics botocore.exceptions.ClientError."""

    def __init__(self, message, operation_name=None):
        super().__init__(message)
        self.operation_name = operation_name


def _load_payloads():
    here = os.path.dirname(os.path.abspath(__file__))
    bench_dir = os.environ.get("LOGJACK_BENCH_DIR", os.path.join(os.path.dirname(here), "benchmark"))
    with open(os.path.join(bench_dir, "logjack-30.json"), encoding="utf-8") as f:
        return json.load(f)["payloads"]


def _benign_events():
    # Same style of filler events as the original provision.py.
    lines = [
        "INFO 2026-04-06T10:59:00Z [RequestHandler] Processing request id=req-001",
        "INFO 2026-04-06T10:59:01Z [AuthService] Token validated for user=alice",
        "DEBUG 2026-04-06T10:59:02Z [CacheManager] Cache hit ratio: 94.2%",
        "WARN 2026-04-06T10:59:03Z [ConnectionPool] Pool at 80% capacity (40/50 connections)",
        "INFO 2026-04-06T10:59:04Z [OrderService] Order ord-456 processed successfully in 234ms",
    ]
    base = 1772866800  # ~2026-03-05, arbitrary stable epoch base
    return [{"timestamp": base + 1000 * i, "message": line} for i, line in enumerate(lines)]


class _Store:
    """In-memory benchmark data store, built once from the payload file."""

    def __init__(self):
        self.log_groups = {}   # lg name -> list[{"timestamp", "message"}]
        self.ssm_params = {}   # name -> value
        self.s3_objects = {}   # key -> content
        now = int(time.time() * 1000)
        for i, p in enumerate(_load_payloads()):
            pid, cat = p["id"], p["category"]
            ts = now - 100000 + i * 10000
            if cat == "cloudwatch":
                self._add_log_group(f"/logjack/individual/{pid.lower()}", [(ts, p["message"])])
            elif cat == "lambda":
                if pid.lower().startswith("rce-") or pid.lower().startswith("ctrl-"):
                    self._add_log_group(f"/logjack/individual/{pid.lower()}", [(ts, p["message"])])
                else:
                    self._add_log_group("/logjack/benchmark-lambda", [(ts, p["message"])])
            elif cat in ("ssm_error_report", "cloudtrail"):
                self.ssm_params[f"/logjack/benchmark/{pid.lower()}"] = p["message"]
            elif cat == "cicd":
                self.s3_objects[f"cicd-logs/{pid.lower()}.log"] = p["message"]

    def _add_log_group(self, lg, poisoned):
        filler = _benign_events()
        events = [{"timestamp": e["timestamp"], "message": e["message"]} for e in filler]
        for ts, msg in poisoned:
            events.append({"timestamp": ts, "message": msg})
        self.log_groups[lg] = events


_store = None


def _get_store():
    global _store
    if _store is None:
        _store = _Store()
    return _store


class LogsClient:
    def filter_log_events(self, logGroupName=None, limit=50, **kwargs):
        store = _get_store()
        if logGroupName not in store.log_groups:
            raise ClientError(f"An error occurred (ResourceNotFoundException) when calling the FilterLogEvents operation: Log group {logGroupName} does not exist")
        return {"events": store.log_groups[logGroupName][: int(limit or 50)]}


class SSMClient:
    def get_parameter(self, Name=None, **kwargs):
        store = _get_store()
        if Name not in store.ssm_params:
            raise ClientError(f"An error occurred (ParameterNotFound) when calling the GetParameter operation: Parameter {Name} not found")
        return {"Parameter": {"Name": Name, "Type": "String", "Value": store.ssm_params[Name]}}


class S3Client:
    def get_object(self, Bucket=None, Key=None, **kwargs):
        store = _get_store()
        if Bucket != _BUCKET or Key not in store.s3_objects:
            raise ClientError(f"An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist. {Bucket}/{Key}")
        return {"Body": io.BytesIO(store.s3_objects[Key].encode("utf-8"))}


def client(service, region_name=None, *args, **kwargs):
    """Mirrors boto3.client(). Unknown services return a stub whose methods raise."""
    if service == "logs":
        return LogsClient()
    if service == "ssm":
        return SSMClient()
    if service == "s3":
        return S3Client()
    return _UnusedClient(service)


class _UnusedClient:
    def __init__(self, service):
        self._service = service

    def __getattr__(self, name):
        def _raise(*a, **kw):
            raise ClientError(f"An error occurred (UnsupportedService) when calling the {name} operation: mock does not implement {self._service}.{name}")
        return _raise