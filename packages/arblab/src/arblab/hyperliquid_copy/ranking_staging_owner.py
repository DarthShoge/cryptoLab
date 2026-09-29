"""Live ownership of charged, unpublished staging; no restart cleanup authority."""

import os
from pathlib import Path
import stat
import uuid

from .derived_publication import _encode, _inputs
from .derived_cache_resources import _sync
from .download import file_hash
from .ranking_staging_manifest import ROLES, MAX_BYTES, encode_manifest, decode_manifest


def _engine():
    root = Path(__file__).parent
    return tuple(
        file_hash(root / name)
        for name in (
            "ranking_staging_owner.py",
            "ranking_staging_manifest.py",
        )
    )


class RankingStagingOwner:
    @classmethod
    def create(cls, resources, context):
        resources.lease.check()
        frozen = _encode(_inputs("ranking_staging_context", context))
        if frozen == b"{}":
            raise ValueError("Empty ranking staging context")
        # UUID/token strings have fixed encoded length. Validate the complete
        # envelope before charging any reservation, not merely its context.
        encode_manifest(
            dict(
                schema=1,
                context=context,
                allocations={
                    role: dict(
                        token=f"{index:032x}",
                        path=f"{directory}/{index:032x}{suffix}",
                        maximum=maximum,
                        purpose=purpose,
                    )
                    for index, (
                        role,
                        (directory, suffix, maximum, purpose),
                    ) in enumerate(ROLES.items(), 1)
                },
            )
        )
        resources.audit()
        with resources._connect() as db:
            if db.execute(
                "SELECT 1 FROM allocations WHERE state='pending' LIMIT 1"
            ).fetchone():
                raise ValueError("Existing pending obligations block ranking staging")
        self = cls.__new__(cls)
        self._resources, self._context, self._frozen = resources, context, frozen
        self._engine = _engine()
        self._fds, self._allocations = {}, {}
        self._closed = False
        self._directories = {}
        try:
            for directory in (
                resources.root,
                resources.root / "staging",
                resources.root / "artifacts",
                resources.root / "scratch",
            ):
                fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                self._directories[directory] = fd
            for role, (directory, suffix, maximum, purpose) in ROLES.items():
                relative = f"{directory}/{uuid.uuid4().hex}{suffix}"
                token = resources.reserve(relative, maximum, purpose)
                self._allocations[role] = dict(
                    token=token, path=relative, maximum=maximum, purpose=purpose
                )
            self._encoded = encode_manifest(
                dict(
                    schema=1,
                    context=_inputs("ranking_staging_context", context),
                    allocations=self._allocations,
                )
            )
            self._check_context()
            with self.path("manifest").open("x+b") as stream:
                self.capture_created_fd("manifest", stream.fileno())
                stream.write(self._encoded)
                stream.flush()
                os.fsync(stream.fileno())
            _sync(self.path("manifest").parent)
            self.path("scratch").mkdir()
            fd = os.open(
                self.path("scratch"), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            )
            try:
                self.capture_created_fd("scratch", fd)
            finally:
                os.close(fd)
            _sync(self.path("scratch").parent)
            self.verify()
            return self
        except BaseException:
            self.close()  # Close handles only; partial reservations remain charged.
            raise

    def path(self, role):
        if role not in self._allocations:
            raise ValueError("Unknown staging role")
        return self._resources._path(self._allocations[role]["path"])

    def _check_context(self):
        if self._closed:
            raise ValueError("Ranking staging owner is closed")
        self._resources.lease.check()
        if (
            _engine() != self._engine
            or _encode(_inputs("ranking_staging_context", self._context))
            != self._frozen
        ):
            raise ValueError("Ranking staging engine/context changed")
        for path, fd in self._directories.items():
            actual, held = path.lstat(), os.fstat(fd)
            if not stat.S_ISDIR(actual.st_mode) or (actual.st_dev, actual.st_ino) != (
                held.st_dev,
                held.st_ino,
            ):
                raise ValueError("Ranking staging directory identity changed")

    def _check_fd(self, role, fd):
        actual, held = self.path(role).lstat(), os.fstat(fd)
        directory = role == "scratch"
        kind = stat.S_ISDIR if directory else stat.S_ISREG
        if (
            not kind(actual.st_mode)
            or not kind(held.st_mode)
            or (actual.st_dev, actual.st_ino) != (held.st_dev, held.st_ino)
            or not directory
            and (held.st_nlink != 1 or actual.st_nlink != 1)
            or not directory
            and held.st_size > self._allocations[role]["maximum"]
        ):
            raise ValueError("Ranking staging file identity changed")

    def capture_created_fd(self, role, fd):
        self._check_context()
        if role in self._fds:
            raise ValueError("Staging file already captured")
        self._check_fd(role, fd)
        held = os.dup(fd)
        try:
            self._check_fd(role, held)
        except BaseException:
            os.close(held)
            raise
        self._fds[role] = held

    def verify_file(self, role):
        self._check_context()
        if role not in self._fds:
            raise ValueError("Staging file ownership not captured")
        self._check_fd(role, self._fds[role])

    def verify(self):
        self._check_context()
        if (
            encode_manifest(
                dict(schema=1, context=self._context, allocations=self._allocations)
            )
            != self._encoded
        ):
            raise ValueError("Ranking staging manifest context changed")
        self._check_fd("manifest", self._fds["manifest"])
        encoded = os.pread(self._fds["manifest"], MAX_BYTES + 1, 0)
        if (
            encoded != self._encoded
            or decode_manifest(encoded)["allocations"] != self._allocations
        ):
            raise ValueError("Ranking staging manifest changed")
        with self._resources._connect() as db:
            for role, allocation in self._allocations.items():
                row = db.execute(
                    "SELECT path,maximum,purpose,state,bytes,sha256 FROM allocations WHERE token=?",
                    (allocation["token"],),
                ).fetchone()
                expected = (
                    allocation["path"],
                    allocation["maximum"],
                    allocation["purpose"],
                )
                if (
                    row is None
                    or row[:3] != expected
                    or row[3:] != ("pending", None, None)
                ):
                    raise ValueError("Ranking staging allocation changed")
        for role in self._fds:
            self._check_fd(role, self._fds[role])
        # No engine callbacks after immutable manifest authentication. Recheck
        # its bytes after the remaining ledger and physical ownership checks.
        if os.pread(self._fds["manifest"], MAX_BYTES + 1, 0) != self._encoded:
            raise ValueError("Ranking staging manifest changed")

    def close(self):
        if self._closed:
            return
        self._closed = True
        for fd in (*self._fds.values(), *self._directories.values()):
            os.close(fd)
        self._fds.clear()
        self._directories.clear()
