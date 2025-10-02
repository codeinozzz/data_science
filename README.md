# Data Science and Machine Learning Vault

The required *Python* version for this project is *3.12.x.*

## About me

My name is Santiago, and I have been at Jala University for three years, where I've worked primarily with JavaScript and Java on various front- and back-end projects.

Throughout my career, I've also implemented several frameworks for each language: Springboot in Java, Angular in TS, and ASP development, managing Express in JS..

## Setup environment

As usual setup your virtual environment:

```
$ python -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip setuptools
$ pip install -r requirements-dev.txt
$ pip install -e .
```

## Basic code compliance

```
$ black .
All done! ✨ 🍰 ✨
X files left unchanged.
$ mypy .
Success: no issues found in X source files
```

## About the CI/CD pipeline

This `monorepo` comes with a pre-configured CI/CD pipeline that is triggered every time a push is made to a **merge request** or when a **merge request** is integrated into the **main** branch.

The pipeline is configured to:

- Execute **code compliance** checks.
- Generate documentation.

> You may add more stages or jobs to the pipeline, but make sure you **do not remove the existing ones**. In addition, make sure you do your best to **keep the pipeline green** at all times.
