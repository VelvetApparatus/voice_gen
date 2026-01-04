#!/usr/bin/env bash

python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

pip install git+https://github.com/huggingface/parler-tts.git

