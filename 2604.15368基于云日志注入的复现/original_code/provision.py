#!/usr/bin/env python3
"""Provision LogJack-30 benchmark payloads into AWS resources."""

import json
import subprocess
import sys
import time

REGION = "us-east-1"

def aws(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ⚠️  {result.stderr.strip()[:200]}")
    return result.returncode == 0

def main():
    with open("benchmark/logjack-30.json") as f:
        data = json.load(f)

    payloads = data["payloads"]
    print(f"Provisioning {len(payloads)} payloads into AWS ({REGION})\n")

    # --- CloudWatch log groups ---
    cw_payloads = [p for p in payloads if p["category"] == "cloudwatch"]
    log_group = "/logjack/benchmark"

    print(f"Creating log group: {log_group}")
    aws(f'aws logs create-log-group --log-group-name {log_group} --region {REGION} 2>/dev/null')

    stream = "benchmark/2026-04-06"
    aws(f'aws logs create-log-stream --log-group-name {log_group} --log-stream-name {stream} --region {REGION} 2>/dev/null')

    # Insert benign entries interspersed with poisoned ones
    benign_logs = [
        "INFO 2026-04-06T10:59:00Z [RequestHandler] Processing request id=req-001",
        "INFO 2026-04-06T10:59:01Z [AuthService] Token validated for user=alice",
        "DEBUG 2026-04-06T10:59:02Z [CacheManager] Cache hit ratio: 94.2%",
        "INFO 2026-04-06T10:59:03Z [MetricsCollector] Published 38 metrics to CloudWatch",
        "WARN 2026-04-06T10:59:04Z [ConnectionPool] Pool at 80% capacity (40/50 connections)",
        "INFO 2026-04-06T10:59:05Z [OrderService] Order ord-456 processed successfully in 234ms",
        "DEBUG 2026-04-06T10:59:06Z [RateLimiter] Request rate: 1,247 req/s (within limits)",
        "INFO 2026-04-06T10:59:07Z [HealthCheck] All upstream dependencies healthy",
        "ERROR 2026-04-06T10:59:08Z [OrderService] Order processing failed: NullPointerException at OrderService.java:142 - order.getPaymentDetails() returned null for orderId=ord-789",
        "WARN 2026-04-06T10:59:09Z [DiskMonitor] Disk usage at 72% on /data volume",
    ]

    now = int(time.time() * 1000)
    events = []

    # Interleave benign and poisoned
    ts = now - 100000
    for i, benign in enumerate(benign_logs):
        events.append({"timestamp": ts, "message": benign})
        ts += 1000
        # Insert a poisoned log after every 2 benign ones
        if i < len(cw_payloads) and i % 2 == 1:
            events.append({"timestamp": ts, "message": cw_payloads[i // 2]["message"]})
            ts += 1000

    # Add remaining poisoned logs
    for p in cw_payloads[len(benign_logs) // 2:]:
        events.append({"timestamp": ts, "message": p["message"]})
        ts += 1000

    # Put in batches of 10 (API limit)
    for i in range(0, len(events), 10):
        batch = events[i:i+10]
        batch_json = json.dumps(batch)
        cmd = f"aws logs put-log-events --log-group-name {log_group} --log-stream-name {stream} --region {REGION} --log-events '{batch_json}'"
        if aws(cmd):
            print(f"  ✓ Inserted log batch {i//10 + 1} ({len(batch)} events)")
        time.sleep(0.5)

    # --- SSM Parameters ---
    ssm_payloads = [p for p in payloads if p["category"] == "ssm_error_report"]
    print(f"\nCreating {len(ssm_payloads)} SSM parameters")

    for p in ssm_payloads:
        name = f"/logjack/benchmark/{p['id'].lower()}"
        value = p["message"].replace("'", "'\\''")
        cmd = f"aws ssm put-parameter --name '{name}' --type String --value '{value}' --overwrite --region {REGION} 2>&1"
        if aws(cmd):
            print(f"  ✓ {name}")

    # --- CI/CD logs as S3 objects ---
    cicd_payloads = [p for p in payloads if p["category"] == "cicd"]
    bucket = "logjack-benchmark-730335305775"

    print(f"\nCreating S3 bucket: {bucket}")
    aws(f'aws s3 mb s3://{bucket} --region {REGION} 2>/dev/null')

    for p in cicd_payloads:
        key = f"cicd-logs/{p['id'].lower()}.log"
        tmpfile = f"/tmp/{p['id']}.log"
        with open(tmpfile, 'w') as f:
            f.write(p["message"])
        if aws(f"aws s3 cp {tmpfile} s3://{bucket}/{key} --region {REGION} 2>&1"):
            print(f"  ✓ s3://{bucket}/{key}")

    # --- CloudTrail-style events as SSM parameters ---
    ct_payloads = [p for p in payloads if p["category"] == "cloudtrail"]
    print(f"\nCreating {len(ct_payloads)} CloudTrail event parameters")

    for p in ct_payloads:
        name = f"/logjack/benchmark/{p['id'].lower()}"
        value = p["message"].replace("'", "'\\''")
        cmd = f"aws ssm put-parameter --name '{name}' --type String --value '{value}' --overwrite --region {REGION} 2>&1"
        if aws(cmd):
            print(f"  ✓ {name}")

    # --- Lambda logs as CloudWatch log group ---
    lambda_payloads = [p for p in payloads if p["category"] == "lambda"]
    lambda_group = "/logjack/benchmark-lambda"

    print(f"\nCreating Lambda log group: {lambda_group}")
    aws(f'aws logs create-log-group --log-group-name {lambda_group} --region {REGION} 2>/dev/null')
    aws(f'aws logs create-log-stream --log-group-name {lambda_group} --log-stream-name "2026/04/06" --region {REGION} 2>/dev/null')

    lambda_events = []
    ts = now - 50000
    for p in lambda_payloads:
        lambda_events.append({"timestamp": ts, "message": p["message"]})
        ts += 5000

    batch_json = json.dumps(lambda_events)
    cmd = f"aws logs put-log-events --log-group-name {lambda_group} --log-stream-name '2026/04/06' --region {REGION} --log-events '{batch_json}'"
    if aws(cmd):
        print(f"  ✓ Inserted {len(lambda_events)} Lambda log events")

    print(f"\n{'='*60}")
    print(f"Provisioning complete: {len(payloads)} payloads across")
    print(f"  - CloudWatch: {len(cw_payloads)} in {log_group}")
    print(f"  - SSM: {len(ssm_payloads)} parameters")
    print(f"  - CI/CD: {len(cicd_payloads)} S3 objects in {bucket}")
    print(f"  - CloudTrail: {len(ct_payloads)} SSM parameters")
    print(f"  - Lambda: {len(lambda_payloads)} in {lambda_group}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
