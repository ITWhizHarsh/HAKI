"""
pytest + Hypothesis configuration for the HAKI Core test suite.

Profiles
--------
default
    Used in normal ``pytest`` runs: 100 examples per property.
ci
    Used in CI: 200 examples, extended deadline.
dev
    Fast feedback during development: 20 examples, relaxed deadline.

Activate a profile via the HYPOTHESIS_PROFILE environment variable:

    HYPOTHESIS_PROFILE=ci pytest

Or call ``settings.load_profile("ci")`` directly in a test module.

Testing conventions (from tasks.md):
- Every property test runs a MINIMUM of 100 iterations.
- Each test is tagged: Feature: haki-personal-ai-assistant, Property N: ...
- Model-backed capabilities are mocked/stubbed so property tests are cheap.
"""

import pytest
from hypothesis import HealthCheck, Phase, settings, Verbosity

# ------------------------------------------------------------------
# Hypothesis settings profiles
# ------------------------------------------------------------------

settings.register_profile(
    "default",
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=2_000,          # 2 s per example
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
)

settings.register_profile(
    "ci",
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=5_000,
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
)

settings.register_profile(
    "dev",
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,            # No deadline in dev mode for easier debugging
    verbosity=Verbosity.verbose,
)

settings.load_profile("default")


# ------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------

@pytest.fixture
def tmp_vault(tmp_path):
    """A fresh temporary vault directory for Memory_Brain tests."""
    vault = tmp_path / "vault"
    vault.mkdir()
    return vault
