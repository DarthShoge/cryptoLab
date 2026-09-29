"""Keep a scratch inode alive until its owner's cleanup checks have finished."""

from contextlib import contextmanager
import os


@contextmanager
def pin_directory(path, identity):
    # A dev/inode pair alone can pass after repeated directory replacement if
    # the filesystem recycles the original inode. This open descriptor prevents
    # reuse; callers must run their path/identity check and cleanup inside it.
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if (info.st_dev, info.st_ino) != identity:
            raise ValueError("Query scratch identity changed while pinning")
        yield
    finally:
        os.close(fd)
