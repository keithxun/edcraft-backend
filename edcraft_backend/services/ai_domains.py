"""Which validator domains users may choose when creating questions."""

import os

from edcraft_validator.domains.domain_registry import available_domains

# Source commit of the validator wheel distributed with this backend.
LIBRARY_REVISION = "8512b16fa6ffe9554f60df92c63bf0db3a9a41c5"

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
