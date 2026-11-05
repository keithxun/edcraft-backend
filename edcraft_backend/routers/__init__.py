"""Router exports."""

from edcraft_backend.routers import (
    ai_authoring_worker,
    assessment_templates,
    assessments,
    folders,
    input_generator,
    question_bank,
    question_generation,
    question_template_banks,
    question_templates,
    questions,
    users,
)

__all__ = [
    "ai_authoring_worker",
    "assessment_templates",
    "assessments",
    "folders",
    "input_generator",
    "question_generation",
    "question_template_banks",
    "question_templates",
    "questions",
    "users",
    "question_bank",
]
