import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="allennlp-skopt-teffland", # Replace with your own username
    version="0.0.1",
    author="Tom Effland",
    author_email="teffland@cs.columbia.edu",
    description="Bayesian hyperparameter tuning with allennlp",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/teffland/allennlp-skopt",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
