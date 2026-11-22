# Deploying AI Authoring with PostgreSQL and Nomad

A teacher submits, reviews, approves and generates questions through the existing AI Authoring API. PostgreSQL holds the job and the exact evidence the teacher approves. In Nomad mode, a separate Docker worker runs the model and the deterministic checks. An API replica can restart while the worker is running. Another replica can receive the result and serve the teacher's review.

## What is stored

The shared `jobs` table holds the job ID, owner, status and Nomad job ID. One row in `authoring_job_data` holds the teacher's request, progress, attempt, review decision, artifact and report. The evidence is stored as bytes with SHA-256 digests. Approval binds to the artifact digest. The existing ownership and approval checks apply to preview and generation. The application's embedding cache uses its own directory.

A queued job stores its provider configuration in an encrypted payload so a different replica can dispatch it. The worker retrieves that configuration from an authenticated callback. Nomad receives a credential scoped to the job and attempt. Protect access to Nomad job specs and API callbacks. Treat database backups as containing teacher material and provider configuration. Provision the same authoring encryption key on every replica and keep it in the installation's secret store. Existing provider connections must remain decryptable with the installation's current configuration.

## First stage — PostgreSQL

1. Arrange a maintenance window and obtain approval for the target environment. Tell teachers that their previous AI Authoring history, including approved templates, will be available only from the archive. Preserve the database, application version and validator needed to read the old store.
2. Pause new authoring submissions, drain running jobs and callbacks, then pause teacher mutations such as approval. Take a database backup. Identify the actual file store used by the old deployment, including records for every owner, and mount it read-only for the archive. The old directory contains `jobs` and `artifacts`.
3. Create the archive in an existing, access-controlled directory outside the store. Use a new destination name. From the backend environment run:

   ```sh
   python -m edcraft_backend.authoring_archive "$OLD_AUTHORING_STORE" "$ARCHIVE_PATH"
   ```

   The command writes a private compressed tar, records file sizes in the tar headers and SHA-256 digests in `SHA256SUMS`, then reads the archived bytes back before publishing the backup. Retain the original volume. Restore the archive into a separate private directory, run `shasum -a 256 -c SHA256SUMS` from that directory, and open a representative approved artifact using its compatible validator revision. Compare its owner, approval digest, seeded question and answer against the original. Restrict archive access and assign a retention owner.
4. Point the new backend at the existing application PostgreSQL database, set `AUTHORING_EXECUTOR=inline`, and run `alembic upgrade head` with the new release. Check the Alembic revision and the following counts immediately after migration, before enabling submissions:

   ```sql
   SELECT count(*) FROM authoring_job_data;
   SELECT count(*) FROM jobs WHERE type = 'ai_authoring';
   ```

   Both counts should be zero on an initial cutover. If either is nonzero, investigate before resuming. Verify that an existing teacher can sign in and access their provider connection. The migration creates the new authoring storage without importing old jobs.
5. Submit one controlled request. Verify that a second replica can see it, a different teacher cannot see it, approval binds to the reviewed digest, and reusing the same seed reproduces the question. Keep PostgreSQL as the source of truth from the first new write.

## Second stage — Nomad

Build the worker from the exact validator commit pinned by the backend's `pyproject.toml`, vendored wheel and `LIBRARY_REVISION`. From a clean checkout of that commit in the AI repository, build a Linux image for the target client architecture:

```sh
docker build --build-arg EDCRAFT_SOURCE_REVISION="$(git rev-parse HEAD)" -t "$WORKER_IMAGE" .
```

Use a registry image digest in deployment and ensure every eligible Nomad client can pull it. The worker starts from the image's own entry point and command. The API registers one Docker batch job per attempt. Nomad's automatic task restarts and rescheduling are disabled for these jobs so a scheduler retry cannot silently repeat a model call.

| Setting | Deployment requirement |
| --- | --- |
| `DATABASE_URL` | All replicas connect to the same migrated database. |
| `AUTHORING_EXECUTOR` | Set to `nomad` for new teacher requests after verifying PostgreSQL. |
| `AUTHORING_NOMAD_CONTAINER_IMAGE` | Registry reference pinned to the verified worker digest. Check the image's revision label against the backend pin. |
| `AUTHORING_NOMAD_CALLBACK_BASE_URL` | HTTPS API address reachable from each worker. Its certificate chain must be trusted inside the image. Preserve the worker's authorization and allocation headers through the ingress. |
| `NOMAD_HOST`, `NOMAD_PORT`, `NOMAD_NAMESPACE`, `NOMAD_DATACENTERS`, `NOMAD_TOKEN` | Use the installation's scheduler endpoint, placement settings and scoped ACL. This executor speaks HTTP to the Nomad API. Use a local trusted agent or protected transport to the cluster. Verify submit, read job, read allocations and stop permissions. |
| `AUTHORING_ENCRYPTION_KEY` | Use the same installation secret across replicas and drains. The backend falls back to the installation JWT secret when this value is absent. Preserve existing provider-key decryption across configuration changes. |
| `AUTHORING_MAX_ACTIVE_JOBS`, `AUTHORING_MAX_ACTIVE_JOBS_PER_USER` | Defaults are four globally and two per user, shared across replicas. The global limit is capped at four. |
| `AUTHORING_JOB_TIMEOUT_SECONDS` | Defaults to 1800 seconds from claim. Set it to accommodate the worker's actual model and checker workload. |

Verify the target cluster in staging with a typed request and a material-based request. Check worker image pull, DNS, trusted TLS, model connectivity, callback delivery through a different replica and deterministic replay after approval. Use an actual authorized staging provider for this last connectivity check. Keep the local rehearsal's fixture CA, private-host override and synthetic credentials confined to local tests.

## Monitoring and rollback

Use this query to tell whether the authoring queue and its occupied worker slots have drained:

```sql
SELECT execution_mode,
       count(*) FILTER (WHERE attempt_id IS NULL AND execution_payload IS NOT NULL) AS queued,
       count(*) FILTER (WHERE attempt_id IS NOT NULL AND attempt_ended_at IS NULL) AS active
FROM authoring_job_data
GROUP BY execution_mode;
```

A completed report may be ready for the teacher before the allocation has exited. Wait for both counts to reach zero and check Nomad allocations. If a node disconnects, the dispatcher preserves capacity while a worker that already obtained its context could still be running. An unavailable scheduler can delay reconciliation. Monitor queue age, overdue attempts, allocation status and callback failures. Retain completed Nomad job identities until reconciliation has finished.

To revert execution, pause new submissions at ingress, keep callbacks reachable, and drain existing work with the original image, secrets and scheduler configuration. Set every replica to `AUTHORING_EXECUTOR=inline`, submit a controlled request, then reopen submissions. A queued job keeps the executor chosen at submission. Keep the PostgreSQL schema and records in place. A storage downgrade deletes new authoring rows. Recovering the old file-backed deployment after new writes requires a separate reconciliation of the new history.

### Recovering an interrupted inline worker

An inline worker runs inside an API process. It can lose its PostgreSQL connection while its model call is still running. In that case the job fails and its slot stays occupied until execution returns. This preserves the four-job limit even through a database interruption. If the entire API process died, an operator must confirm the recorded process or container has stopped before releasing its slot.

Pause new submissions. Identify the affected job and attempt, read the recorded host and process ID, and confirm the original process or container has exited. If the process could still be running, keep the slot occupied. Once its exit is confirmed, use a transaction scoped to that exact failed attempt. The following uses `psql` variables set by the operator from the verified record:

```sql
BEGIN;
SELECT id
FROM jobs
WHERE id = :'job_id'::uuid AND type = 'ai_authoring'
FOR UPDATE;

UPDATE authoring_job_data d
SET attempt_ended_at = clock_timestamp(), execution_payload = NULL
FROM jobs j
WHERE j.id = d.job_id
  AND j.type = 'ai_authoring' AND j.status = 'failed'
  AND d.execution_mode = 'inline' AND d.attempt_ended_at IS NULL
  AND d.job_id = :'job_id'::uuid AND d.attempt_id = :'attempt_id'::uuid
  AND d.worker_claim_id = :'confirmed_worker'
  AND (d.record->'error'->>'code') IN ('SERVER_RESTARTED', 'AUTHORING_TIMEOUT')
RETURNING d.job_id, d.attempt_id, d.attempt_ended_at;
```

Verify the returned IDs before `COMMIT`. If the row does not match, `ROLLBACK` and investigate. This operation releases a slot and leaves the teacher's failed job in history.

## Local evidence

The local rehearsal exercised PostgreSQL, real single-node Nomad allocations, worker-to-API HTTPS, cross-replica approval, and an in-flight API restart with a controlled provider fixture. It did not exercise the deployment cluster's ACL, registry, routing or model service. Keep that cluster check and production approval as separate rollout steps.
