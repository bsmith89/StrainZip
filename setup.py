import setuptools

# Long description and versioning code copied with modifications from
# <https://github.com/blackjax-devs/blackjax/blob/main/setup.py>

with open("README.md") as f:
    long_description = f.read()


def get_version(path):
    """Get the package's version number.
    We fetch the version  number from the `__version__` variable located in the
    package root's `__init__.py` file. This way there is only a single source
    of truth for the package's version number.
    """
    with open(path) as f:
        for line in f:
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
        else:
            raise RuntimeError("Unable to find version string.")


setuptools.setup(
    name="StrainZip",
    version=get_version("src/strainzip/__init__.py"),
    long_description=long_description,
    description="Unified co-assembly and pseudo-alignment",
    url="http://github.com/bsmith89/StrainZip",
    author="Byron J. Smith",
    author_email="me@byronjsmith.com",
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        # "graph-tool>=2.59",  # This can't be installed with pip...womp womp.
        "pandas>=2",
        "scipy>=1",
    ],
    dependency_links=[],
    entry_points={"console_scripts": ["strainzip = strainzip.__main__:main"]},
    zip_safe=False,
    long_description_content_type="text/markdown",
)
