"""Which validator domains users may choose when creating questions."""

import os

from edcraft_validator.domains.domain_registry import available_domains

# Source commit of the validator wheel distributed with this backend.
LIBRARY_REVISION = "eb86a932cb20ec7436e60bcc690d3ecea5cc05a5"

ENABLED_DOMAINS_ENV = "AI_AUTHORING_ENABLED_DOMAINS"
DEFAULT_ENABLED_DOMAINS = ("code", "math", "biology", "chemistry", "physics")


def enabled_domains() -> tuple[str, ...]:
    """Registered domains that are switched on, in registry order.

    A domain merged into the validator stays hidden until it is listed here, so
    re-pinning the validator cannot expose a domain before its evaluation.
    """
    raw = os.environ.get(ENABLED_DOMAINS_ENV)
    allowed = (
        DEFAULT_ENABLED_DOMAINS
        if raw is None
        else tuple(name.strip() for name in raw.split(",") if name.strip())
    )
    return tuple(name for name in available_domains() if name in allowed)
