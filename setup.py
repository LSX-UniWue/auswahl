import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nirSelect",
    version="0.1.0",
    author="Florian Buckermann",
    author_email="buckermann@informatik.uni-wuerzburg.de",
    description="Collection of Feature Selection Methods for Near-Infrared Spectroscopy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mcFloskel/nirSelect",
    project_urls={
        "Bug Tracker": "https://github.com/mcFloskel/nirSelect/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    include_package_data=True,
    install_requires=["numpy", "scikit-learn"],
    python_requires=">=3.7",
)