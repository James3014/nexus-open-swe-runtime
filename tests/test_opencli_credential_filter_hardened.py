import pytest

from nexus_open_swe_runtime.opencli_web_model import OpenCLIWebChatModel

FILTERED = (
    "GITHUB_TOKEN",
    "github_token",
    "gh_token",
    "Token",
    "SOME_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "MY_SECRET",
    "DB_PASSWORD",
    "DB_PASSWD",
    "SERVICE_CREDENTIAL",
    "PROVIDER_TOKEN",
    "OPENCLI_TOKEN",
    "OPENCLI_API_KEY",
    "STRIPE_KEY",
    "stripe_key",
    "AWS_ACCESS_KEY_ID",
    "AWS_REGION",
    "PRIVATE_KEY",
    "SSH_PRIVATE_KEY",
    "AUTHORIZATION",
    "BASIC_AUTH",
    "AUTH_HEADER",
    "SESSION_COOKIE",
    "COOKIE",
    "GITLAB_PAT",
    "SSH_AUTH_SOCK",
    "KUBECONFIG",
    "DOCKER_AUTH_CONFIG",
    "DATABASE_URL",
    "GH_OPENCLI_HOST",
)
KEPT = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/home/kept",
    "LANG": "C.UTF-8",
    "TMPDIR": "/tmp/kept",
    "OPENCLI_PROFILE": "kept-profile",
    "OPENCLI_KEEP_ME": "kept-value",
    "AUTHOR_NAME": "kept-author",
    "PROVIDER_NAME": "kept-provider",
    "MONKEY_COUNT": "3",
}


def test_opencli_child_environment_uses_hardened_credential_filter(
    monkeypatch: pytest.MonkeyPatch,
):
    for name in FILTERED:
        monkeypatch.setenv(name, f"leak-{name.lower()}")
    for name, value in KEPT.items():
        monkeypatch.setenv(name, value)

    env = OpenCLIWebChatModel(model="chatgpt")._environment()

    leaked = sorted(name for name in FILTERED if name in env)
    assert not leaked
    for name, value in KEPT.items():
        assert env.get(name) == value, name
