"""Explicit worker-local verification policy; persistent engine keys stay intact.

Install the selected checksum implementation at the existing import boundaries.
The policy and this implementation's actual hashes are frozen in the experiment
provenance. This runs only inside an owned, single-job worker; the API coordinator
continues using full verification. Nothing is installed in a running old worker.
"""

from contextlib import contextmanager
import sys

from arblab.hyperliquid_copy import download
from .lab_file_hash_session import FileHashSession

POLICIES = ("full", "guarded-session-v1")


def _replace(original, replacement):
    for name, module in tuple(sys.modules.items()):
        if name.startswith("arblab.hyperliquid_copy.") and module is not None:
            for attribute, value in tuple(vars(module).items()):
                if value is original:
                    setattr(module, attribute, replacement)


@contextmanager
def verification_session(policy):
    if policy not in POLICIES:
        raise ValueError("Invalid verification policy")
    if policy == "full":
        yield None
        return
    original = download.file_hash
    if isinstance(original, FileHashSession):
        raise ValueError("Nested verification sessions are not supported")
    session = FileHashSession()
    _replace(original, session)
    try:
        yield session
    finally:
        # Include modules imported during execution as well as initial imports.
        _replace(session, original)
        session.entries.clear()
