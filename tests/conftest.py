import pytest
import requests

API_URL = "http://localhost:5005/"
MODEL_URL = "http://localhost:8501/v1/models/interests-model"


def _is_up(url):
    try:
        requests.get(url, timeout=2).raise_for_status()
    except requests.RequestException:
        return False
    return True


@pytest.fixture(scope="session")
def _servers_running():
    """Fail with a readable message instead of a ConnectionError traceback."""
    if not _is_up(MODEL_URL):
        pytest.fail(
            "TF Serving is not reachable on port 8501 — "
            "run `make train`, then `make serve-model`.",
            pytrace=False,
        )

    if not _is_up(API_URL):
        pytest.fail(
            "The Flask API is not reachable on port 5005 — "
            "run `make serve-api` (see task 2).",
            pytrace=False,
        )
