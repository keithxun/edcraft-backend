# EdCraft Backend

FastAPI backend for the **EdCraft**, providing APIs for question generation and content management.

## 🚀 Overview

EdCraft Backend exposes the EdCraft Engine via a REST API and handles database operations. It handles authentication, data management, and async job processing for question generation.

## 🧱 Tech Stack

* Python 3.12+
* FastAPI
* PostgreSQL
* SQLAlchemy
* Alembic
* Nomad
* Docker

## ⚡ Quick Start

### 1. Clone & install

```bash
git clone https://github.com/edcraft-org/edcraft-backend.git
cd edcraft-backend
uv sync
```

Requires [`uv`](https://github.com/astral-sh/uv)


### 2. Setup environment

```bash
cp .env.example .env.development
```

Update database and secrets in `.env.development`.


### 3. Start services

```bash
# Start database
make db-up

# Start Nomad
make nomad

# Run backend
make dev
```


### 4. Access API

* API: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## 🧪 Testing

```bash
# Start test db
make db-test

# Run tests
make test
```

## 📁 Project Structure

```
edcraft_backend/
├── routers/        # API endpoints
├── services/       # Business logic
├── repositories/   # DB access layer
├── models/         # SQLAlchemy models
├── schemas/        # Pydantic schemas
├── config/         # Settings
├── security/       # Auth utilities
├── oauth/          # OAuth integration
├── executors/      # External jobs (Nomad)
├── main.py         # App entrypoint
alembic/            # DB migrations
tests/              # Test suite
worker/             # Nomad job worker
```

## 🗄️ Database

Run migrations:

```bash
uv run alembic upgrade head
```

Create a migration:

```bash
uv run alembic revision --autogenerate -m "message"
```

Access database (dev):

```bash
docker exec -it edcraft-postgres psql -U edcraft_dev -d edcraft
```

For details about the schema, see: [Database Documentation](docs/database.md)

## 🔐 Authentication

* JWT (access and refresh tokens in httpOnly cookies)
* OAuth via GitHub

### Basic flow

1. `POST /auth/signup`
2. `POST /auth/login`
3. Use authenticated endpoints
4. `POST /auth/refresh` (token rotation)

## ⚙️ Async Jobs (Nomad)

Long-running endpoints (e.g. question generation) are **asynchronous**.

### 🧠 How it works

1. Client sends request

2. Backend:
   * generates a callback token
   * submits a Nomad job

3. API responds immediately:

```json
{ "job_id": "...", "status_url": "/jobs/{id}" }
```

4. Nomad:

   * spins up a worker container
   * injects worker code at runtime
   * executes the task

5. Worker sends result back:

```
POST /jobs/callback/{token}
```

6. Client polls:

```
GET /jobs/{job_id}
```

### Example response

```json
{
  "status": "completed",
  "result": {...}
}
```

## 🤖 AI Template Authoring

Issues 48-50 add owner-scoped AI authoring endpoints under `/ai-authoring`.
The backend uses the pinned `edcraft-validator` workflow in a bounded
in-process runner (one active job, ten waiting) and persists state as files,
not database rows. Run exactly one backend process (one Uvicorn worker/replica)
against each authoring storage volume: the process-local lock, admission
counters and startup reconciliation are not multi-process safe.

Configure at least one server-side model profile in `.env.development`:

```bash
AUTHORING_MODEL_PROFILES=[{"id":"default","label":"Default code author","provider":"soclaas","model":"x-test-2"}]
AUTHORING_STORAGE_DIR=.data/authoring
```

Server profiles are the legacy fallback. Users can instead bring their own
OpenAI or SocLaas key under **Settings → Model connections**. The backend
stores only an encrypted key (never plaintext) and returns safe metadata such
as the key's last four characters. Keys are decrypted in memory for a single
job and are never written to job records or logs. Set a stable
`AUTHORING_ENCRYPTION_KEY` in production (generate with
`openssl rand -hex 32`); when omitted, the installation `JWT_SECRET` is used.
Changing the encryption key makes existing connections undecryptable.

Provider credentials for server profiles remain in the provider-specific
environment. They are never returned by the API. Docker Compose mounts the
durable `authoring_data` volume at `/app/.data`, and the image runs as the
non-root `edcraft` user. Files are written atomically with `os.replace`;
startup marks interrupted queued/running jobs failed with `SERVER_RESTARTED`
instead of replaying them.

Advisory difficulty and relevance checks use the same model and credentials as
the selected authoring model, so BYOK jobs are billed entirely to the caller's
key. `EDCRAFT_JUDGE_REASONING_EFFORT` and `EDCRAFT_RELEVANCE_THRESHOLD` still
configure those checks; `EDCRAFT_JUDGE_PROVIDER`/`MODEL` only affect direct
CLI or MCP use of `edcraft-validator`.

### Check authoring pilot

The private polynomial check pilot is exposed under
`/ai-authoring/check-pilot`:

- `POST /templates`, `GET /templates`, `GET /templates/{template_id}`
- `GET /checks`, `GET /checks/{check_id}`
- `POST /templates/{template_id}/approval`
- `POST /templates/{template_id}/generate`

All routes require the existing user authentication and only expose the
signed-in owner's records. `POST /templates` accepts exactly one existing
author source: a configured `AUTHORING_MODEL_PROFILES` entry
(`model_profile`) or an owner-scoped BYOK `connection_id` with `model`. The
same resolved provider serves the AI application's check-author and
template-author calls and is closed after each request. Responses are the
AI-owned `edcraft-validator` models, documented in OpenAPI.

Records live beside legacy authoring state at
`$AUTHORING_STORAGE_DIR/check-pilot` and are not database rows. Keep the same
single-backend-process and durable-volume constraint as authoring. Bounds:

```bash
CHECK_PILOT_MAX_CONCURRENT_REQUESTS=2       # workers per process
CHECK_PILOT_ADMISSION_TIMEOUT_SECONDS=5     # wait for a free worker
CHECK_PILOT_REQUEST_TIMEOUT_SECONDS=180     # wall clock per request
```

Generation quantity is additionally capped at `min(AUTHORING_MAX_QUANTITY, 50)`.
Admission exhaustion returns 429, timed-out workers 504, and provider/storage
failures 503. The authenticated `GET /ai-authoring/capabilities` response
publishes `check_pilot_timeout_seconds`, the sum of the configured admission
and request bounds, so clients can choose a timeout with a positive transport
margin instead of hardcoding the execution default. This route requires the
pinned `edcraft-validator` revision
`724e02aec72a67e80b8f9ab7927b73679c93eef0`. Integration order is the AI commit
first, then this pinned backend, then the frontend; no publication or deploy is
part of this change.

## 🖥️ WSL2 Nomad Setup

If you're using **WSL2**, if you are facing issues using nomad, follow the instructions in this section.

Docker containers cannot access `localhost` on the host.

### One-time setup

```bash
docker network create --subnet 192.168.100.0/24 edcraft-network
docker tag edcraft-backend:latest edcraft-backend:local
```

Set in `.env.development`:

```
NOMAD_CONTAINER_IMAGE=edcraft-backend:local
```

---

### Each session

1. Start Nomad:

```bash
make nomad
```

2. Update Nomad host IP (if not working):

```bash
sed -i "s/NOMAD_HOST=.*/NOMAD_HOST=$(ip addr show eth0 | grep 'inet ' | awk '{print $2}' | cut -d/ -f1)/" .env.development
```

3. Start backend:

```bash
make docker-up
```

## Enhancements

* Add rate limiting to protect APIs and improve system stability
* Add pagination support for data retrieval endpoints
* Add caching to improve performance and reduce repeated computations
* Add unit tests to improve test coverage alongside existing integration tests
* Implement proper logging for debugging, monitoring, and error tracking

## 📄 License

MIT License © 2025 EdCraft Team
