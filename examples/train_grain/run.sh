#!/bin/bash
cd "$(dirname "$0")"
python prepare_data.py
python run.py
