"""Python setup.py for mm_dream package"""
import io
import os
from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("mm_dream", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="mm_dream",
    version=read("mm_dream", "VERSION"),
    description="Awesome mm_dream created by jkobject",
    url="https://github.com/jkobject/MM_DREAM/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="jkobject",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        "console_scripts": ["mm_dream = mm_dream.__main__:main"]
    },
    extras_require={"test": read_requirements("requirements-test.txt")},
)
