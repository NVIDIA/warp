# Use this file to bootstrap packman into your Python environment (3.7.x). Simply
# add the path by doing sys.insert to where packmanconf.py is located and then execute:
#
# >>> import packmanconf
# >>> packmanconf.init()
#
# It will use the configured remote(s) and the version of packman in the same folder,
# giving you full access to the packman API via the following module
#
# >> import packmanapi
# >> dir(packmanapi)

import os
import platform
import sys


def init():
    """Call this function to initialize the packman configuration.

    Calls to the packman API will work after successfully calling this function.

    Note:
        This function only needs to be called once during the execution of your
        program. Calling it repeatedly is harmless but wasteful.
        Compatibility with your Python interpreter is checked and upon failure
        the function will report what is required.

    Example:
        >>> import packmanconf
        >>> packmanconf.init()
        >>> import packmanapi
        >>> packmanapi.set_verbosity_level(packmanapi.VERBOSITY_HIGH)
    """
    major = sys.version_info[0]
    minor = sys.version_info[1]
    if major != 3 or minor != 7:
        raise RuntimeError(
            f"This version of packman requires Python 3.7.x, but {major}.{minor} was provided"
        )
    conf_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ["PM_INSTALL_PATH"] = conf_dir
    packages_root = get_packages_root(conf_dir)
    version = get_version(conf_dir)
    module_dir = get_module_dir(conf_dir, packages_root, version)
    sys.path.insert(1, module_dir)


def get_packages_root(conf_dir: str) -> str:
    root = os.getenv("PM_PACKAGES_ROOT")
    if not root:
        platform_name = platform.system()
        if platform_name == "Windows":
            drive, _ = os.path.splitdrive(conf_dir)
            root = os.path.join(drive, "packman-repo")
        elif platform_name == "Darwin":
            # macOS
            root = os.path.join(
                os.path.expanduser("~"), "/Library/Application Support/packman-cache"
            )
        elif platform_name == "Linux":
            try:
                cache_root = os.environ["XDG_HOME_CACHE"]
            except KeyError:
                cache_root = os.path.join(os.path.expanduser("~"), ".cache")
            return os.path.join(cache_root, "packman")
        else:
            raise RuntimeError(f"Unsupported platform '{platform_name}'")
    # make sure the path exists:
    os.makedirs(root, exist_ok=True)
    return root


def get_module_dir(conf_dir, packages_root: str, version: str) -> str:
    module_dir = os.path.join(packages_root, "packman-common", version)
    if not os.path.exists(module_dir):
        import tempfile

        tf = tempfile.NamedTemporaryFile(delete=False)
        target_name = tf.name
        tf.close()
        url = f"http://bootstrap.packman.nvidia.com/packman-common@{version}.zip"
        print(f"Downloading '{url}' ...")
        import urllib.request

        urllib.request.urlretrieve(url, target_name)
        from importlib.machinery import SourceFileLoader

        # import module from path provided
        script_path = os.path.join(conf_dir, "bootstrap", "install_package.py")
        ip = SourceFileLoader("install_package", script_path).load_module()
        print("Unpacking ...")
        ip.install_package(target_name, module_dir)
        os.unlink(tf.name)
    return module_dir


def get_version(conf_dir: str):
    path = os.path.join(conf_dir, "packman")
    if not os.path.exists(path):  # in dev repo fallback
        path += ".sh"
    with open(path, "rt", encoding="utf8") as launch_file:
        for line in launch_file.readlines():
            if line.startswith("PM_PACKMAN_VERSION"):
                _, value = line.split("=")
                return value.strip()
    raise RuntimeError(f"Unable to find 'PM_PACKMAN_VERSION' in '{path}'")
