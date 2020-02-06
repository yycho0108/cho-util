#!/usr/bin/env bash
git push
rm -rf dist
python3 setup.py sdist
twine upload dist/*
