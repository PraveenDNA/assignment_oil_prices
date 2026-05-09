# AWS Deployment Design — Brent Crude Stress-Test

## How a risk analyst invokes this on demand

A risk analyst opens an internal web portal (or calls a REST endpoint directly) with parameters:
start date, number of paths, horizon, and seed.  Within minutes they receive a pre-signed S3 link
to a self-contained `report.html`.

---

## Architecture

```
Analyst / CI
    │
    ▼
API Gateway (REST)
    │  POST /run  { start, n_paths, horizon, seed }
    ▼
Lambda (orchestrator)
    │  — validates input
    │  — publishes job message
    ▼
SQS FIFO Queue
    │  — deduplication by (start, n_paths, horizon, seed)
    ▼
ECS Fargate Task  (pulled from ECR image)
    │  — runs python run.py --start ... --n-paths ...
    │  — writes reports/report.html
    ▼
S3 Bucket  (versioned, private)
    │  — s3://brent-stress-artefacts/reports/YYYY-MM-DD-HHMMSS-{seed}/report.html
    │  — s3://brent-stress-artefacts/models/{seed}/params.json
    ▼
Lambda (post-run)
    │  — generates pre-signed URL (TTL 4 hours)
    │  — writes job status to DynamoDB
    ▼
API Gateway response → analyst receives URL
```

---

## Service justifications

| Service | Role | Justification |
|---------|------|---------------|
| **API Gateway (REST)** | Invocation path | Managed, no infra; supports IAM auth and API key throttling; cheap at low usage |
| **Lambda (orchestrator)** | Input validation + job dispatch | Millisecond invocation, $0/idle; 15-min limit is fine for orchestration only |
| **SQS FIFO** | Job queue | Deduplication prevents identical run requests from spawning duplicate Fargate tasks; decouples invocation from execution |
| **ECS Fargate** | Model execution | The full pipeline takes 1–5 minutes; Lambda's 15-min limit is sufficient but Fargate is cleaner for CPU-bound workloads and allows >10 GB memory if needed |
| **ECR** | Container registry | Docker image (Python 3.11 + arch + scipy + matplotlib) pinned by digest; no dependency drift |
| **S3 (versioned)** | Artefact storage | Reports and model params stored per run; versioning allows audit trail; lifecycle policy expires objects after 90 days |
| **DynamoDB** | Job status | Stores `{job_id, status, s3_key, created_at, params}` for polling and audit |
| **Secrets Manager** | Secrets | Any future API keys (e.g. premium data provider) stored here; IAM task role grants read-only access; no hardcoded credentials |
| **IAM task roles** | Identity | Fargate task role: `s3:PutObject` on the artefacts bucket only. Orchestrator Lambda role: `sqs:SendMessage` + `ecs:RunTask`. Principle of least privilege throughout |
| **CloudWatch Logs** | Observability | Structured JSON logs from the pipeline (`INFO` level); log group per task family |
| **CloudWatch Metrics + Alarms** | Observability | Custom metric `StressTestDuration`; alarm → SNS → email/PagerDuty if p99 duration > 10 min or error rate > 5% |
| **X-Ray** | Distributed tracing | Traces API Gateway → Lambda → SQS → Fargate; useful for latency debugging |

---

## Cost estimate (current usage: ~10 runs/day)

| Item | Monthly estimate |
|------|-----------------|
| Fargate (0.5 vCPU, 2 GB, ~3 min/run, 300 runs/mo) | ~$3 |
| S3 storage (~50 MB/report × 300 reports) | ~$0.04 |
| ECR storage (~1 GB image) | ~$0.10 |
| Lambda invocations (600/mo) | < $0.01 |
| SQS, DynamoDB, CloudWatch | < $1 |
| **Total** | **~$5/month** |

---

## Identity and secrets

- All IAM roles follow least-privilege.  No standing `AdministratorAccess`.
- Fargate task role has **only** `s3:PutObject` (artefacts bucket, scoped by prefix) and
  `secretsmanager:GetSecretValue` (specific secret ARN).
- No credentials in container image or environment variables.  All secrets via Secrets Manager
  at runtime.
- API Gateway requires AWS SigV4 signing; internal analysts access via SSO-federated role.

---

## Observability

- Every pipeline run emits structured logs: `{ "job_id": "...", "n_obs": 2500, "persistence": 0.98, "n_pass": 6, "duration_s": 142 }`.
- CloudWatch dashboard: run count, median duration, PASS rate, error count.
- Alarm: if any run ends with `n_pass < total_metrics`, a CloudWatch Alarm fires → SNS → team channel.

---

## At 100× usage (~1,000 runs/day)

Three changes are required:

1. **Concurrency**: SQS → Fargate auto-scaling (target 50 concurrent tasks).  Consider
   Fargate Spot for a 70% cost reduction on non-latency-sensitive jobs (analysts happy to wait
   5 min vs 3 min for a $2 vs $6 monthly saving per 1,000 runs).

2. **Cost management**: At $50–$100/month, add S3 lifecycle rules (transition to Glacier after
   30 days), and DynamoDB on-demand billing with TTL to expire old job records automatically.

3. **Queue management**: Add SQS visibility timeout tuning and a Dead-Letter Queue (DLQ) for
   failed tasks, so retries are automatic and failures are surfaced without manual inspection.

A diagram is available on request.
