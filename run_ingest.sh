#!/bin/bash
# run_ingest.sh — فهرسة الملفات بالبيئة الصحيحة مع تثبيت أي ناقص تلقائيًا
set -e
cd "$(dirname "$0")"

PY=python3
if [ -x ".venv/bin/python3" ]; then
    PY=".venv/bin/python3"
    "$PY" -m pip install -q -r requirements.txt
else
    pip3 install -q -r requirements.txt --break-system-packages
fi

exec "$PY" ingest.py
