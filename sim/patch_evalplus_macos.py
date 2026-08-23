"""
Idempotent workaround for an upstream evalplus bug on macOS.

`reliability_guard` sets RLIMIT_AS and RLIMIT_DATA unconditionally. On macOS those raise
`ValueError: current limit exceeds maximum limit`, which kills every check subprocess, so
MBPP+ grading returns 0/N for everything and silently looks like "the model failed". The
function already special-cases Darwin for RLIMIT_STACK; this extends the same treatment to
the other two. Because evalplus spawns (not forks) its check subprocesses, a monkeypatch in
the parent cannot reach them -- the installed module has to be patched.

Run once:  python3 sim/patch_evalplus_macos.py
"""

import inspect
import sys

import evalplus.eval.utils as u

MARKER = "# rrl-macos-rlimit-patch"

OLD = """        resource.setrlimit(
            resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
        )"""

NEW = f"""        {MARKER}: macOS rejects these limits outright; a failure here must not kill
        # the check subprocess, which would read as "the candidate failed every test".
        try:
            resource.setrlimit(
                resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
            )
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(
                resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
            )
        except (ValueError, OSError):
            pass"""


def main() -> int:
    path = inspect.getsourcefile(u)
    src = open(path).read()
    if MARKER in src:
        print(f"already patched: {path}")
        return 0
    if OLD not in src:
        print(f"ERROR: expected block not found in {path}; evalplus version may differ")
        return 1
    open(path, "w").write(src.replace(OLD, NEW))
    print(f"patched {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
