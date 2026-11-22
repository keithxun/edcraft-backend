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

PDF uploads in the materials-to-practice pilot require Poppler's `pdftotext` on `PATH`.
Install `poppler` with Homebrew or `poppler-utils` on Debian or Ubuntu. The backend Docker
image includes it.

Fetch the pinned CPU model once to enable semantic topic retrieval.

```bash
uv run python -m edcraft_backend.services.material_embeddings
```

This downloads about 24 MB and verifies the asset checksums.
[MiniLM model files](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/826711e54e001c83835913827a843d8dd0a1def9)

By default setup writes to
`.data/material-embedding`. Set `EDCRAFT_MATERIAL_EMBEDDING_DIR` for both setup and the
running backend to use a different directory, such as a mounted volume. Each backend
process runs one semantic search at a time with two ONNX CPU threads and caches up to
1,024 page hashes and their passage vectors in RAM. The first search loads the model and
encodes each page. A restart clears the cache. For long pages, the ranker samples up to
eight passages across the page and uses the highest cosine similarity to the topic.
Selected-page and keyword retrieval work before the model is installed.

### Practice generation from course materials

Choose a subject and topic, review the selected lecture pages, and optionally select
an exam excerpt as a style reference. Continue to AI Authoring to submit the reviewed
excerpts with the teacher's request. The model proposes a reusable question and the
subject checker verifies the answer and distractors. Preview the resulting template
and its evidence before approving it for seeded question generation.

The materials API handles PDF extraction, topic suggestions and retrieval. Its retired
`/materials/propose` and `/materials/generate` endpoints return HTTP 410. The earlier
fixed-exercise generator is available in Git history at the November 20 production
revision, `6722868`.

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


### Local five-subject demo

Place the frontend checkout beside this backend, named `edcraft-frontend`. For Compose, configure `DATABASE_URL` in `.env.development` to use the `postgres` service hostname and the same credentials as `POSTGRES_USER`, `POSTGRES_PASSWORD` and `POSTGRES_DB`. For a local password-login demo, set `EMAIL_ENABLED=false` in that file so you can create an active demo account without SMTP. For host-based development the database hostname is normally `localhost`. The backend includes a checksum-locked validator wheel built from the Git revision recorded under `vendor/`. The source is maintained in the validator repository on `codex/domain-seam`.

```sh
docker compose --profile demo up --build -d
docker compose --profile demo exec backend python -m edcraft_backend.services.material_embeddings
```

Open http://localhost:3000, create a local account, sign in and select **AI Authoring → From module materials**. The second command enables local semantic retrieval. Page selection and keyword retrieval work independently.

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

The existing `/ai-authoring` endpoints store owner-scoped jobs in PostgreSQL.
The shared `jobs` table tracks execution. The linked `authoring_job_data` row
preserves the reviewed request, exact artifact and report bytes, their hashes,
the teacher's decision, and seeded generation metadata. Apply Alembic migrations
before starting the API.

Difficulty and relevance inherit the selected question model and its connection.
Teachers can choose a different model for either check in Advanced settings.
Setting `EDCRAFT_JUDGE_MODEL` explicitly supplies a deployment default for
relevance alone. The teacher's selection takes precedence. The SoCLaaS
`x-test-2` difficulty check uses low reasoning effort when its first
classification needs a follow-up assessment.

Before submission, the form checks whether each judge returns token probabilities
for its answer. The probes use short synthetic prompts and reuse recent results.
If a judge is incompatible, the form names the fallback for the affected check.
The question-writing model stays selected. Set `AUTHORING_DECISION_FALLBACK_PROVIDER`
and `AUTHORING_DECISION_FALLBACK_MODEL` to change the server fallback from
`soclaas` / `x-test-2`. Configure the corresponding server credential. A temporary
provider failure is shown as unverified. An unusable decision during a job can
try the fallback once. The report records the model that judged the question.

The default `AUTHORING_EXECUTOR=inline` runs the validator locally. Set
`AUTHORING_EXECUTOR=nomad` to schedule one pinned authoring worker per attempt.
Both modes use database-backed admission. With the default settings at most four
jobs run globally and two per owner. Users configure model keys under **Settings →
Model connections**. Replicas must share a stable `AUTHORING_ENCRYPTION_KEY` to
decrypt provider credentials and outstanding job context.

The [AI Authoring deployment runbook](docs/ai-authoring-deployment.md) covers the
archive, empty database cutover, Nomad worker settings, smoke tests and rollback.
Compose uses a separate volume for the optional material embedding cache.
Repopulate it with the local demo setup command if needed.

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
