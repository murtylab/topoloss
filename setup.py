import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="topoloss",
    version="0.0.0",
    description="topoloss",
    author="Mayukh Deb, Mainak Deb, N. A. Apurva Ratan Murty",
    author_email="mayukhmainak2000@gmail.com, mayukh@gatech.edu",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/toponets/topoloss",
    packages=["topoloss"],
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
