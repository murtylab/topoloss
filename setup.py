import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

"""
release checklist:
1. update version on `setup.py`
2. run tests with this command `pytest -vvx tests/`
4. commit changes (`setup.py`) and push
5. make release on PyPI. Run the following commands:
    5.1 `python3 setup.py sdist bdist_wheel`
    5.2 (optional) `python3 -m pip install --user --upgrade twine`
    5.3 `python3 -m twine upload dist/*`
6. make a new release on github with the latest version
    6.1 git tag v0.0.1 ## or your version name
    6.2 git push origin v0.0.1
"""

setuptools.setup(
    name="topoloss",
    version="0.0.2",
    description="topoloss",
    author="Mayukh Deb, Mainak Deb, N. Apurva Ratan Murty",
    author_email="mayukhmainak2000@gmail.com, mayukh@gatech.edu",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/toponets/topoloss",
    packages=setuptools.find_packages(),
    install_requires=[
        "einops",
        "torchtyping",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
