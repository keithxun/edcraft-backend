# Nomad AI authoring execution (backend note)

This is the short implementation note for the backend half of the flag-gated
Nomad authoring mode. The full operations runbook lives outside this repository.

## Enablement

`AUTHORING_EXECUTOR` defaults to `inline`; the in-process thread runner and all
public `/ai-authoring` responses stay unchanged. Set `AUTHORING_EXECUTOR=nomad`
to submit one Nomad batch allocation per admitted authoring job.

| Variable | Default | Purpose |
| --- | --- | --- |
| `AUTHORING_EXECUTOR` | `inline` | `inline` or `nomad`. |
| `AUTHORING_WORKER_TOKEN_TTL_SECONDS` | `7200` | Callback token lifetime. |
| `AUTHORING_JOB_TIMEOUT_SECONDS` | `1800` | Runner wait before timeout. |
| `AUTHORING_NOMAD_CONTAINER_IMAGE` | required in Nomad mode | Authoring worker image; no builder-image fallback. |
| `AUTHORING_NOMAD_CALLBACK_BASE_URL` | `NOMAD_CALLBACK_BASE_URL` | Worker-to-backend URL. |
| `AUTHORING_NOMAD_FORCE_PULL` | `false` | Force image pull per job. |

`AUTHORING_MAX_ACTIVE_JOBS`, per-user limits, queueing and admission are
unchanged. The store and pending futures are process-local, so the backend
remains a single replica.

`AUTHORING_NOMAD_CONTAINER_IMAGE` is mandatory in Nomad mode. It deliberately
does **not** fall back to `NOMAD_CONTAINER_IMAGE` (the builder engine image):
that image has no `edcraft_validator.worker` module. When the authoring image is
unset, `POST /ai-authoring/jobs` returns `422` before admission, the store is
untouched, and no Nomad job is submitted.

## Flow

1. Admission reserves capacity exactly as in inline mode.
2. The service generates `secrets.token_urlsafe(32)` and persists only
   `sha256(token).hexdigest()` plus an expiry on the job record.
3. The runner registers a pending future, then submits a Nomad batch job running
   `python -m edcraft_validator.worker`.
4. The worker fetches `GET /ai-authoring/worker/{token}/context`, which returns
   the command plus decrypted role providers with `Cache-Control: no-store`.
5. Progress, the final `AuthoringResult`, or a fatal worker error arrive at the
   other token-authenticated endpoints.
6. The runner resolves the future, persists the one result (or timeout /
   failure), and releases admission exactly once.

The job spec carries only `EDCRAFT_PARAMS_B64` (command JSON), callback URLs,
the pinned library revision and the job id. Model API keys and registry
credentials never appear in the Nomad spec, stored job record or logs.

## Semantics

- Unknown or expired tokens return `401`.
- A result with the wrong library revision returns `422`.
- A job that already finished returns `409`, except an identical duplicate
  result which is idempotent (`204`).
- `needs_review` remains `state="completed"` with
  `result.error.code="TEMPLATE_CHECKS_FAILED"`.
- A timeout stops the Nomad job, fails the job with `AUTHORING_TIMEOUT`, and
  releases the admission slot.
- Shutdown cancels pending callback waits and best-effort stops each job.

## Rollback

Set `AUTHORING_EXECUTOR=inline` (or unset it) and restart. Stored records keep
`format_version=1` and all worker-channel fields are optional, so no data
migration is needed.
