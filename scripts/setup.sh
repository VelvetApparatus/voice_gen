#!/usr/bin/env bash

python -m venv venv
source venv/bin/activate

pip install git+https://github.com/huggingface/parler-tts.git

pip install -r requirements.txt
