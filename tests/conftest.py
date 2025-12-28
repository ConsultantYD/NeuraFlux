from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure():
    """
    Ensure tests import the workspace version of the package.

    Some environments add `/app` to `sys.path` (via a `.pth` file). In this repo, `/app`
    may contain an older checkout, which can shadow the workspace code under
    `/workspaces/NeuraFlux`.
    """

    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)

    # Drop other checkouts that can shadow the workspace code.
    sys.path = [p for p in sys.path if p not in {"/app"}]

    # Ensure repo root is at the front of sys.path.
    sys.path = [p for p in sys.path if p != repo_root_str]
    sys.path.insert(0, repo_root_str)
