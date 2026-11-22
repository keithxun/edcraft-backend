# Database Documentation

This document provides a detailed overview of the database design and schema used.

## 🧱 Overview

- Database: PostgreSQL
- ORM: SQLAlchemy
- Migrations: Alembic

Set `DATABASE_ECHO=true` to log SQL statements. SQLAlchemy hides bind parameters such as authoring prompts and reports in its statement logs and formatted database errors.

## 📊 Schema Diagram


```mermaid
erDiagram
    users ||--o{ refresh_tokens : "has"
    users ||--o{ oauth_accounts : "has"
    users ||--o{ folders : "owns"
    users ||--o{ assessments : "owns"
    users ||--o{ assessment_templates : "owns"
    users ||--o{ questions : "owns"
    users ||--o{ question_templates : "owns"
    users ||--o{ question_template_banks : "owns"
    users ||--o{ resource_collaborators : "has"
    users ||--o{ one_time_tokens : "has"
    users ||--o{ jobs : "owns"

    folders ||--o{ folders : "contains (parent-child)"
    folders ||--o{ assessments : "contains"
    folders ||--o{ assessment_templates : "contains"
    folders ||--o{ question_banks : "contains"
    folders ||--o{ question_template_banks : "contains"

    assessments ||--o{ questions : "contains (FK)"
    question_banks ||--o{ questions : "contains (FK)"
    questions ||--o{ questions : "linked_from (self-ref)"

    assessment_templates ||--o{ question_templates : "contains (FK)"
    question_template_banks ||--o{ question_templates : "contains (FK)"
    question_templates ||--o{ question_templates : "linked_from (self-ref)"

    question_templates ||--o{ questions : "generates"
    question_templates ||--o{ target_elements : "has"

    questions ||--o| mcq_data : "MCQ data"
    questions ||--o| mrq_data : "MRQ data"
    questions ||--o| short_answer_data : "short answer data"

    jobs ||--o{ job_tokens : "has"

    users {
        uuid id PK
        string email
        string name
        string password_hash "nullable (OAuth users)"
        bool is_active
        datetime created_at
        datetime updated_at
        datetime deleted_at "soft delete"
    }

    refresh_tokens {
        uuid id PK
        uuid user_id FK
        string token_hash UK
        datetime expires_at
        bool is_revoked
        string ip_address "nullable"
        string user_agent "nullable"
    }

    oauth_accounts {
        uuid id PK
        uuid user_id FK
        string provider
        string provider_user_id
    }

    folders {
        uuid id PK
        uuid owner_id FK
        uuid parent_id FK "nullable"
        string name
        text description "nullable"
        datetime created_at
        datetime updated_at
        datetime deleted_at "soft delete"
    }

    assessments {
        uuid id PK
        uuid owner_id FK
        uuid folder_id FK
        string title
        text description
        string visibility
        datetime created_at
        datetime updated_at
        datetime deleted_at "soft delete"
    }

    question_banks {
        uuid id PK
        uuid owner_id FK
        uuid folder_id FK
        string title
        text description
        string visibility
        datetime created_at
        datetime updated_at
        datetime deleted_at "soft delete"
    }

    questions {
        uuid id PK
        uuid owner_id FK
        uuid template_id FK "nullable, SET NULL"
        uuid assessment_id FK "nullable, SET NULL"
        uuid question_bank_id FK "nullable, SET NULL"
        uuid linked_from_question_id FK "nullable, SET NULL (self-ref)"
        int order "nullable, unique per assessment"
        string question_type
        text question_text
        datetime created_at
        datetime updated_at
        datetime deleted_at "soft delete"
    }

    question_template_banks {
        uuid id PK
        uuid owner_id FK
        uuid folder_id FK
        string title
        text description
        string visibility
        datetime created_at
        datetime updated_at
        datetime deleted_at "soft delete"
    }

    assessment_templates {
        uuid id PK
        uuid owner_id FK
        uuid folder_id FK
        string title
        text description
        string visibility
        datetime created_at
        datetime updated_at
        datetime deleted_at "soft delete"
    }

    question_templates {
        uuid id PK
        uuid owner_id FK
        uuid assessment_template_id FK "nullable, SET NULL"
        uuid question_template_bank_id FK "nullable, SET NULL"
        uuid linked_from_template_id FK "nullable, SET NULL (self-ref)"
        int order "nullable, unique per assessment_template"
        string question_type
        text question_text_template
        string text_template_type
        text description "nullable"
        text code
        string entry_function
        int num_distractors
        string output_type
        json input_data_config "nullable"
        json code_info "nullable"
        datetime created_at
        datetime updated_at
        datetime deleted_at "soft delete"
    }

    mcq_data {
        uuid question_id FK
        array options
        int correct_index
    }

    mrq_data {
        uuid question_id FK
        array options
        array correct_indices
    }

    short_answer_data {
        uuid question_id FK
        text correct_answer
    }

    resource_collaborators {
        uuid id PK
        string resource_type
        uuid resource_id
        uuid user_id FK
        string role
        datetime added_at
    }

    jobs {
        uuid id PK
        string type
        string status
        string nomad_job_id "nullable"
        text result_json "nullable"
        text error_message "nullable"
        uuid user_id FK "nullable, SET NULL"
        datetime created_at
        datetime completed_at "nullable"
    }

    job_tokens {
        string token PK
        uuid job_id FK
        datetime consumed_at "nullable"
        bool revoked
        datetime created_at
    }

    one_time_tokens {
        uuid id PK
        uuid user_id FK
        string token_hash UK
        string token_type
        datetime expires_at
        bool is_used
        datetime created_at
    }

    target_elements {
        uuid template_id FK
        int order PK
        string element_type
        array id_list
        string name "nullable"
        int line_number "nullable"
        string modifier "nullable"
        array argument_keys "nullable"
    }
```


## 🧩 Core Concepts

### 1. Ownership Model

* Every resource has an **owner (`user_id`)**
* Access control is handled via:

  * ownership
  * collaborators (see API layer)

---

### 2. Folder Hierarchy

* Tree structure (self-referencing)
* Each user has a **root folder**
* Resources live inside folders:

  * assessments
  * question banks
  * templates

---

### 3. Questions & Templates

#### Questions

* Stored in assessments and question banks

#### Question Templates (generation logic)

* Used to generate questions dynamically

---

### 4. Copy-on-Link Pattern

When linking:

* A **new copy is created**
* Original reference is stored
* Copies are independently editable

---

### 5. Ordering

For ordered collections (e.g. assessments):

* 0-indexed
* Always contiguous (0,1,2,...)
* Insert shifts items down
* Deletion re-normalizes order

---

### 6. Async Jobs

The `jobs` table tracks background work:

* `queued → running → completed/failed`
* Stores:

  * result JSON
  * error message
* Linked to Nomad workers via `job_tokens`

## 🧪 Testing Database

* Separate test DB
* Runs in isolated transactions
* Automatically rolled back after tests

## 📌 Notes

* Full schema definitions live in `edcraft_backend/models/`
* Migration history lives in `alembic/versions/`
