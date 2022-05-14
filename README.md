# owkin_mm_dream

[![codecov](https://codecov.io/gh/jkobject/owkin_MM_DREAM/branch/main/graph/badge.svg?token=owkin_MM_DREAM_token_here)](https://codecov.io/gh/jkobject/owkin_MM_DREAM)
[![CI](https://github.com/jkobject/owkin_MM_DREAM/actions/workflows/main.yml/badge.svg)](https://github.com/jkobject/owkin_MM_DREAM/actions/workflows/main.yml)

Awesome Cancer recurrence prediction of Multiple Myeloma DREAM challenge v2 created by jkobject.

## Information

This pipeline shows various simple ML models used to predict patient with Multiple Myeloma at high risk of relapse from transcriptomics profiles.
From a comprehensive litterature review I came up with a set of genes, genesets and clinical information to use as predictors.

3 Models are shown together with some metrics and used to define strengths and weaknesses of each.

The default model selected for prediction is a logistic regression with elasticnet penalty, an l1_ratio of 1.0 and a constraint term of 0.2.

Information about the literature review, my work schedule and process during this 10h mini-project are available in `NOTEBOOK.md`

## Install it

```bash
git clone https://github.com/jkobject/owkin_MM_DREAM.git
cd owkin_MM_DREAM
pip install -e .
```

A dockerized version is also available

```bash
docker pull jkobject/mm_dream
```

## Usage

### showcase

```py
#look at example.py !
from owkin_mm_dream import main

clf = main(syn_login, syn_password)
```

or

```bash
python -m owkin_mm_dream $syn_login $syn_password
#or
docker run -it jkobject/mm_dream $syn_login $syn_password
```

### predict

```py
#look at example.py for more info!
from owkin_mm_dream import main

clf = main(syn_login, syn_password)
res = clf.predict(X, Y)
```

or

```bash
python owkin_mm_dream $syn_login $syn_password $rna_path $clinical_path
#or
docker run -it jkobject/mm_dream $syn_login $syn_password $rna_path $clinical_path
#extract the file out with cp
docker cp <containerId>:/app/owkin/owkin_mm_dream  /host/path/target
```

the results will be in `new_predictions.csv`


## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## Structure

Lets take a look at the structure of this template:

```text
├── Containerfile            # The file to build a container using buildah or docker
├── CONTRIBUTING.md          # Onboarding instructions for new contributors
├── docs                     # Documentation site (add more .md files here)
│   └── index.md             # The index page for the docs site
├── .github                  # Github metadata for repository
│   ├── release_message.sh   # A script to generate a release message
│   └── workflows            # The CI pipeline for Github Actions
├── .gitignore               # A list of files to ignore when pushing to Github
├── LICENSE                  # The license for the project
├── Makefile                 # A collection of utilities to manage the project
├── MANIFEST.in              # A list of files to include in a package
├── mkdocs.yml               # Configuration for documentation site
├── owkin_mm_dream             # The main python package for the project
│   ├── base.py              # The base module for the project
│   ├── __init__.py          # This tells Python that this is a package
│   ├── __main__.py          # The entry point for the project
│   └── VERSION              # The version for the project is kept in a static file
├── README.md                # The main readme for the project
├── setup.py                 # The setup.py file for installing and packaging the project
├── requirements.txt         # An empty file to hold the requirements for the project
├── requirements-test.txt    # List of requirements for testing and devlopment
├── setup.py                 # The setup.py file for installing and packaging the project
└── tests                    # Unit tests for the project (add mote tests files here)
    ├── conftest.py          # Configuration, hooks and fixtures for pytest
    ├── __init__.py          # This tells Python that this is a test package
    └── test_base.py         # The base test case for the project
```

## The Makefile

All the utilities for the template and project are on the Makefile

```bash
❯ make
Usage: make <target>

Targets:
help:             ## Show the help.
install:          ## Install the project in dev mode.
fmt:              ## Format code using black & isort.
lint:             ## Run pep8, black, mypy linters.
test: lint        ## Run tests and generate coverage report.
watch:            ## Run tests on every change.
clean:            ## Clean unused files.
virtualenv:       ## Create a virtual environment.
release:          ## Create a new tag for release.
docs:             ## Build the documentation.
switch-to-poetry: ## Switch to poetry package manager.
init:             ## Initialize the project based on an application template.
```
