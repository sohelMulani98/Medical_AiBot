#!/bin/bash
#creating a basic project template structure

mkdir -p src
mkdir -p reserch

touch src/__init__.py
touch src/helpers.py
touch src/prompts.py

touch .env
touch requirements.txt
touch setup.py
touch app.py

touch reserch/trials.ipynb

echo "Project template structure created successfully."