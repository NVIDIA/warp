import os
import sys
import io
import contextlib
import packmanapi

REPO_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
REPO_DEPS_FILE = os.path.join(REPO_ROOT, "deps/repo-deps.packman.xml")


def bootstrap():
    """
    Bootstrap all omni.repo modules.

    Pull with packman from repo.packman.xml and add them all to python sys.path to enable importing.
    """
    #with contextlib.redirect_stdout(io.StringIO()):
    deps = packmanapi.pull(REPO_DEPS_FILE)
    for dep_path in deps.values():
        if dep_path not in sys.path:
            sys.path.append(dep_path)


if __name__ == "__main__":
    bootstrap()
    import omni.repo.man

    omni.repo.man.main(REPO_ROOT)
