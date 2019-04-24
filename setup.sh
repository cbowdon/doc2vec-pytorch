#!/usr/bin/env bash

python3 -m venv venv &&
    venv/bin/pip install --upgrade pip &&
    venv/bin/pip install -r requirements.txt &&
    venv/bin/python3 -m spacy download en_core_web_sm
