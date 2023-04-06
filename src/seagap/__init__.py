from pkg_resources import DistributionNotFound, get_distribution

package_name = __name__

try:
    VERSION = get_distribution(package_name).version
except DistributionNotFound:  # pragma: no cover
    try:
        from .version import version as VERSION  # noqa
    except ImportError:  # pragma: no cover
        raise ImportError(
            "Failed to find (autogenerated) version.py. "
            "This might be because you are installing from GitHub's tarballs, "
            "use the PyPI ones."
        )
__version__ = VERSION
