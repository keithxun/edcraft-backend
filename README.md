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

Choose a subject, review the retrieved lecture excerpt, and optionally select one complete
exam subquestion. The subject narrows the eligible operations. The lecture and exam text
determine whether a supported operation applies. Glossary recall and table lookup are
available across subjects when the selected material supports them.

| Subject | Supported operations from course materials |
| --- | --- |
| Programming | Supported C output traces for nested loops, pointer swaps, and recursive reductions |
| Mathematics | Ratio and limit comparison tests, steepest ascent, and classification of separable quartic critical points |
| Physics | Steady-current resistor power with resistance in Ω and current in A, and Hooke’s law force magnitude with stiffness in N/m and extension in m within the proportional limit |
| Chemistry | Smallest whole-number coefficients for a curated acid–base neutralisation equation written with both reactants and products |
| Biology | Unconditional dominant or recessive phenotype probability in a complete-dominance monohybrid pea-plant cross, with both canonical parent genotypes and the allele pair stated |

For the science operations, include a direct statement of the solving rule in the selected
lecture text and either the complete exam task or one matching `Example:` or `Worked example:`
in the lecture.
For example, a Physics lecture can explain `P = I²R` and include this exercise:

> Worked example: A resistor of resistance 10 Ω carries a steady current of 0.4 A.
> What power does it dissipate?

Each science batch samples five variants from a finite set of checked classroom cases. It
keeps the requested operation, excludes the selected example, and rechecks the key and
distinct options for every generated question. Repeating the same input and seed reproduces
the batch for the same generator version. The materials workflow supports the operations in
the table. The separate authoring workflow uses the broader syllabus catalogues.

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
