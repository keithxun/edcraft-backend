"""Dependency injection setup for repositories and services."""

from functools import lru_cache
from typing import Annotated
from uuid import UUID

from fastapi import Cookie, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from edcraft_backend.config import settings
from edcraft_backend.database import get_db
from edcraft_backend.executors.nomad import NomadExecutor
from edcraft_backend.models.user import User
from edcraft_backend.repositories.assessment_repository import AssessmentRepository
from edcraft_backend.repositories.assessment_template_repository import (
    AssessmentTemplateRepository,
)
from edcraft_backend.repositories.folder_repository import FolderRepository
from edcraft_backend.repositories.job_repository import (
    JobRepository,
    JobTokenRepository,
)
from edcraft_backend.repositories.oauth_account_repository import OAuthAccountRepository
from edcraft_backend.repositories.one_time_token_repository import (
    OneTimeTokenRepository,
)
from edcraft_backend.repositories.provider_connection_repository import (
    ProviderConnectionRepository,
)
from edcraft_backend.repositories.question_bank_repository import QuestionBankRepository
from edcraft_backend.repositories.question_repository import QuestionRepository
from edcraft_backend.repositories.question_template_bank_repository import (
    QuestionTemplateBankRepository,
)
from edcraft_backend.repositories.question_template_repository import (
    QuestionTemplateRepository,
)
from edcraft_backend.repositories.refresh_token_repository import RefreshTokenRepository
from edcraft_backend.repositories.resource_collaborator_repository import (
    ResourceCollaboratorRepository,
)
from edcraft_backend.repositories.target_element_repository import (
    TargetElementRepository,
)
from edcraft_backend.repositories.user_repository import UserRepository
from edcraft_backend.security import decode_token
from edcraft_backend.services.ai_authoring_service import AiAuthoringService
from edcraft_backend.services.assessment_service import AssessmentService
from edcraft_backend.services.assessment_template_service import (
    AssessmentTemplateService,
)
from edcraft_backend.services.auth_service import AuthService
from edcraft_backend.services.check_pilot_service import CheckPilotService
from edcraft_backend.services.collaboration_service import CollaborationService
from edcraft_backend.services.email_service import EmailService
from edcraft_backend.services.folder_service import FolderService
from edcraft_backend.services.job_service import JobService
from edcraft_backend.services.oauth_service import OAuthService
from edcraft_backend.services.post_processing_service import PostProcessingService
from edcraft_backend.services.provider_connection_service import (
    ProviderConnectionService,
)
from edcraft_backend.services.provider_secrets import ProviderSecretCipher
from edcraft_backend.services.question_bank_service import QuestionBankService
from edcraft_backend.services.question_service import QuestionService
from edcraft_backend.services.question_template_bank_service import (
    QuestionTemplateBankService,
)
from edcraft_backend.services.question_template_service import QuestionTemplateService
from edcraft_backend.services.user_service import UserService


# Repository dependencies
def get_user_repository(db: AsyncSession = Depends(get_db)) -> UserRepository:
    """Get UserRepository instance."""
    return UserRepository(db)


def get_folder_repository(db: AsyncSession = Depends(get_db)) -> FolderRepository:
    """Get FolderRepository instance."""
    return FolderRepository(db)


def get_question_repository(db: AsyncSession = Depends(get_db)) -> QuestionRepository:
    """Get QuestionRepository instance."""
    return QuestionRepository(db)


def get_question_template_repository(
    db: AsyncSession = Depends(get_db),
) -> QuestionTemplateRepository:
    """Get QuestionTemplateRepository instance."""
    return QuestionTemplateRepository(db)


def get_assessment_repository(
    db: AsyncSession = Depends(get_db),
) -> AssessmentRepository:
    """Get AssessmentRepository instance."""
    return AssessmentRepository(db)


def get_question_bank_repository(
    db: AsyncSession = Depends(get_db),
) -> QuestionBankRepository:
    """Get QuestionBankRepository instance."""
    return QuestionBankRepository(db)


def get_question_template_bank_repository(
    db: AsyncSession = Depends(get_db),
) -> QuestionTemplateBankRepository:
    """Get QuestionTemplateBankRepository instance."""
    return QuestionTemplateBankRepository(db)


def get_assessment_template_repository(
    db: AsyncSession = Depends(get_db),
) -> AssessmentTemplateRepository:
    """Get AssessmentTemplateRepository instance."""
    return AssessmentTemplateRepository(db)


def get_target_element_repository(
    db: AsyncSession = Depends(get_db),
) -> TargetElementRepository:
    """Get TargetElementRepository instance."""
    return TargetElementRepository(db)


def get_refresh_token_repository(
    db: AsyncSession = Depends(get_db),
) -> RefreshTokenRepository:
    """Get RefreshTokenRepository instance."""
    return RefreshTokenRepository(db)


def get_oauth_account_repository(
    db: AsyncSession = Depends(get_db),
) -> OAuthAccountRepository:
    """Get OAuthAccountRepository instance."""
    return OAuthAccountRepository(db)


def get_one_time_token_repository(
    db: AsyncSession = Depends(get_db),
) -> OneTimeTokenRepository:
    """Get OneTimeTokenRepository instance."""
    return OneTimeTokenRepository(db)


def get_resource_collaborator_repository(
    db: AsyncSession = Depends(get_db),
) -> ResourceCollaboratorRepository:
    """Get ResourceCollaboratorRepository instance."""
    return ResourceCollaboratorRepository(db)


# Service dependencies
def get_email_service() -> EmailService:
    """Get EmailService instance."""
    return EmailService()


def get_question_service(
    question_repo: QuestionRepository = Depends(get_question_repository),
    collaborator_repo: ResourceCollaboratorRepository = Depends(
        get_resource_collaborator_repository
    ),
) -> QuestionService:
    """Get QuestionService instance."""
    return QuestionService(question_repo, collaborator_repo)


def get_question_template_service(
    template_repo: QuestionTemplateRepository = Depends(
        get_question_template_repository
    ),
    target_element_repo: TargetElementRepository = Depends(
        get_target_element_repository
    ),
    collaborator_repo: ResourceCollaboratorRepository = Depends(
        get_resource_collaborator_repository
    ),
) -> QuestionTemplateService:
    """Get QuestionTemplateService instance."""
    return QuestionTemplateService(
        template_repo, target_element_repo, collaborator_repo
    )


def get_folder_service(
    folder_repo: FolderRepository = Depends(get_folder_repository),
    assessment_repo: AssessmentRepository = Depends(get_assessment_repository),
    question_bank_repo: QuestionBankRepository = Depends(get_question_bank_repository),
    assessment_template_repo: AssessmentTemplateRepository = Depends(
        get_assessment_template_repository
    ),
    question_svc: QuestionService = Depends(get_question_service),
    question_template_svc: QuestionTemplateService = Depends(
        get_question_template_service
    ),
    qt_bank_repo: QuestionTemplateBankRepository = Depends(
        get_question_template_bank_repository
    ),
) -> FolderService:
    """Get FolderService instance."""
    return FolderService(
        folder_repo,
        assessment_repo,
        question_bank_repo,
        assessment_template_repo,
        qt_bank_repo,
        question_svc,
        question_template_svc,
    )


def get_user_service(
    user_repo: UserRepository = Depends(get_user_repository),
    folder_svc: FolderService = Depends(get_folder_service),
) -> UserService:
    """Get UserService instance."""
    return UserService(user_repo, folder_svc)


def get_collaboration_service(
    collaborator_repo: ResourceCollaboratorRepository = Depends(
        get_resource_collaborator_repository
    ),
    user_repo: UserRepository = Depends(get_user_repository),
    folder_svc: FolderService = Depends(get_folder_service),
    assessment_repo: AssessmentRepository = Depends(get_assessment_repository),
    question_bank_repo: QuestionBankRepository = Depends(get_question_bank_repository),
    qt_bank_repo: QuestionTemplateBankRepository = Depends(
        get_question_template_bank_repository
    ),
    assessment_template_repo: AssessmentTemplateRepository = Depends(
        get_assessment_template_repository
    ),
) -> CollaborationService:
    """Get CollaborationService instance."""
    return CollaborationService(
        collaborator_repo,
        user_repo,
        folder_svc,
        assessment_repo,
        question_bank_repo,
        qt_bank_repo,
        assessment_template_repo,
    )


def get_assessment_service(
    assessment_repo: AssessmentRepository = Depends(get_assessment_repository),
    folder_svc: FolderService = Depends(get_folder_service),
    question_svc: QuestionService = Depends(get_question_service),
    user_repo: UserRepository = Depends(get_user_repository),
    collaboration_svc: CollaborationService = Depends(get_collaboration_service),
) -> AssessmentService:
    """Get AssessmentService instance."""
    return AssessmentService(
        assessment_repo,
        folder_svc,
        question_svc,
        user_repo,
        collaboration_svc,
    )


def get_question_bank_service(
    question_bank_repo: QuestionBankRepository = Depends(get_question_bank_repository),
    folder_svc: FolderService = Depends(get_folder_service),
    question_svc: QuestionService = Depends(get_question_service),
    collaboration_svc: CollaborationService = Depends(get_collaboration_service),
) -> QuestionBankService:
    """Get QuestionBankService instance."""
    return QuestionBankService(
        question_bank_repo,
        folder_svc,
        question_svc,
        collaboration_svc,
    )


def get_question_template_bank_service(
    qt_bank_repo: QuestionTemplateBankRepository = Depends(
        get_question_template_bank_repository
    ),
    folder_svc: FolderService = Depends(get_folder_service),
    qt_repo: QuestionTemplateRepository = Depends(get_question_template_repository),
    question_template_svc: QuestionTemplateService = Depends(
        get_question_template_service
    ),
    collaboration_svc: CollaborationService = Depends(get_collaboration_service),
) -> QuestionTemplateBankService:
    """Get QuestionTemplateBankService instance."""
    return QuestionTemplateBankService(
        qt_bank_repo, folder_svc, qt_repo, question_template_svc, collaboration_svc
    )


def get_assessment_template_service(
    template_repo: AssessmentTemplateRepository = Depends(
        get_assessment_template_repository
    ),
    folder_svc: FolderService = Depends(get_folder_service),
    question_template_svc: QuestionTemplateService = Depends(
        get_question_template_service
    ),
    qt_repo: QuestionTemplateRepository = Depends(get_question_template_repository),
    collaboration_svc: CollaborationService = Depends(get_collaboration_service),
) -> AssessmentTemplateService:
    """Get AssessmentTemplateService instance."""
    return AssessmentTemplateService(
        template_repo,
        folder_svc,
        question_template_svc,
        qt_repo,
        collaboration_svc,
    )


def get_auth_service(
    user_repo: UserRepository = Depends(get_user_repository),
    refresh_token_repo: RefreshTokenRepository = Depends(get_refresh_token_repository),
    one_time_token_repo: OneTimeTokenRepository = Depends(
        get_one_time_token_repository
    ),
    folder_svc: FolderService = Depends(get_folder_service),
    email_svc: EmailService = Depends(get_email_service),
) -> AuthService:
    """Get AuthService instance."""
    return AuthService(
        user_repo, refresh_token_repo, one_time_token_repo, folder_svc, email_svc
    )


def get_oauth_service(
    user_repo: UserRepository = Depends(get_user_repository),
    oauth_account_repo: OAuthAccountRepository = Depends(get_oauth_account_repository),
    auth_svc: AuthService = Depends(get_auth_service),
    folder_svc: FolderService = Depends(get_folder_service),
) -> OAuthService:
    """Get OAuthService instance."""
    return OAuthService(user_repo, oauth_account_repo, auth_svc, folder_svc)


async def get_post_processing_service(
    assessment_svc: AssessmentService = Depends(get_assessment_service),
) -> PostProcessingService:
    """Get PostProcessingService instance."""
    return PostProcessingService(assessment_svc)


async def get_current_user(
    user_repo: UserRepository = Depends(get_user_repository),
    access_token: str | None = Cookie(None),
) -> User:
    """Resolve the authenticated user from the access_token httpOnly cookie."""

    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated"
        )
    try:
        payload = decode_token(access_token)
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type"
            )
        user = await user_repo.get_by_id(UUID(payload["sub"]))
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="User inactive"
            )
        return user
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        ) from e


async def get_current_user_optional(
    user_repo: UserRepository = Depends(get_user_repository),
    access_token: str | None = Cookie(None),
) -> User | None:
    """Resolve authenticated user from cookie, or None if not authenticated.

    For endpoints accessible to both authenticated and unauthenticated users.
    """
    if not access_token:
        return None

    try:
        payload = decode_token(access_token)
        if payload.get("type") != "access":
            return None
        user = await user_repo.get_by_id(UUID(payload["sub"]))
        if not user or not user.is_active:
            return None
        return user
    except Exception:
        return None


# Type aliases for cleaner router signatures
ResourceCollaboratorRepositoryDep = Annotated[
    ResourceCollaboratorRepository, Depends(get_resource_collaborator_repository)
]
UserServiceDep = Annotated[UserService, Depends(get_user_service)]
FolderServiceDep = Annotated[FolderService, Depends(get_folder_service)]
QuestionServiceDep = Annotated[QuestionService, Depends(get_question_service)]
QuestionTemplateServiceDep = Annotated[
    QuestionTemplateService,
    Depends(get_question_template_service),
]
AssessmentServiceDep = Annotated[AssessmentService, Depends(get_assessment_service)]
QuestionBankServiceDep = Annotated[
    QuestionBankService, Depends(get_question_bank_service)
]
QuestionTemplateBankServiceDep = Annotated[
    QuestionTemplateBankService,
    Depends(get_question_template_bank_service),
]
AssessmentTemplateServiceDep = Annotated[
    AssessmentTemplateService,
    Depends(get_assessment_template_service),
]
CollaborationServiceDep = Annotated[
    CollaborationService, Depends(get_collaboration_service)
]
AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
OAuthServiceDep = Annotated[OAuthService, Depends(get_oauth_service)]
CurrentUserDep = Annotated[User, Depends(get_current_user)]
CurrentUserOptionalDep = Annotated[User | None, Depends(get_current_user_optional)]

# Repository type aliases
UserRepositoryDep = Annotated[UserRepository, Depends(get_user_repository)]
FolderRepositoryDep = Annotated[FolderRepository, Depends(get_folder_repository)]
QuestionRepositoryDep = Annotated[QuestionRepository, Depends(get_question_repository)]
QuestionTemplateRepositoryDep = Annotated[
    QuestionTemplateRepository,
    Depends(get_question_template_repository),
]
AssessmentRepositoryDep = Annotated[
    AssessmentRepository, Depends(get_assessment_repository)
]
QuestionBankRepositoryDep = Annotated[
    QuestionBankRepository, Depends(get_question_bank_repository)
]
QuestionTemplateBankRepositoryDep = Annotated[
    QuestionTemplateBankRepository,
    Depends(get_question_template_bank_repository),
]
AssessmentTemplateRepositoryDep = Annotated[
    AssessmentTemplateRepository,
    Depends(get_assessment_template_repository),
]
OneTimeTokenRepositoryDep = Annotated[
    OneTimeTokenRepository,
    Depends(get_one_time_token_repository),
]
EmailServiceDep = Annotated[EmailService, Depends(get_email_service)]


# Nomad
def get_job_repository(db: AsyncSession = Depends(get_db)) -> JobRepository:
    """Get JobRepository instance."""
    return JobRepository(db)


def get_job_token_repository(db: AsyncSession = Depends(get_db)) -> JobTokenRepository:
    """Get JobTokenRepository instance."""
    return JobTokenRepository(db)


def get_nomad_executor() -> NomadExecutor:
    """Get NomadExecutor instance."""
    return NomadExecutor()


def get_job_service(
    job_repo: JobRepository = Depends(get_job_repository),
    job_token_repo: JobTokenRepository = Depends(get_job_token_repository),
    executor: NomadExecutor = Depends(get_nomad_executor),
    post_processing_svc: PostProcessingService = Depends(get_post_processing_service),
) -> JobService:
    """Get JobService instance."""
    return JobService(job_repo, job_token_repo, executor, post_processing_svc)


JobServiceDep = Annotated[JobService, Depends(get_job_service)]


@lru_cache
def get_ai_authoring_service() -> AiAuthoringService:
    """Get the process-wide file-backed AI authoring service."""
    return AiAuthoringService(settings.authoring)


AiAuthoringServiceDep = Annotated[
    AiAuthoringService, Depends(get_ai_authoring_service)
]


@lru_cache
def get_check_pilot_service() -> CheckPilotService:
    """Get the process-wide check-pilot integration service."""
    return CheckPilotService(
        authoring=settings.authoring,
        settings=settings.check_pilot,
    )


CheckPilotServiceDep = Annotated[
    CheckPilotService, Depends(get_check_pilot_service)
]


def get_provider_connection_repository(
    db: AsyncSession = Depends(get_db),
) -> ProviderConnectionRepository:
    """Get the owner-scoped provider connection repository."""
    return ProviderConnectionRepository(db)


def get_provider_secret_cipher() -> ProviderSecretCipher:
    """Use the dedicated authoring key, falling back to the JWT secret."""
    configured = settings.authoring.encryption_key
    secret = configured.get_secret_value() if configured is not None else settings.jwt.secret
    return ProviderSecretCipher(secret)


def get_provider_connection_service(
    repository: ProviderConnectionRepository = Depends(get_provider_connection_repository),
    cipher: ProviderSecretCipher = Depends(get_provider_secret_cipher),
) -> ProviderConnectionService:
    """Get the provider connection service for the current request."""
    return ProviderConnectionService(repository, cipher)


ProviderConnectionServiceDep = Annotated[
    ProviderConnectionService, Depends(get_provider_connection_service)
]
