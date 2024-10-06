import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="topoloss",
    version="0.2.0",
    description="topoloss",
    author="Mayukh Deb",
    author_email="mayukh@gatech.edu, mayukhmainak2000@gmail.com",
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
