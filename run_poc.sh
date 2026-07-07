#!/bin/bash
# run_poc.sh — الإعداد الكامل بأمر واحد: Ollama + النماذج + الفهرسة + الواجهة
set -e
cd "$(dirname "$0")"

echo "==> 1) التحقق من Ollama"
if ! curl -s --max-time 3 http://localhost:11434/api/tags > /dev/null; then
    echo "Ollama غير شغّال — جاري تشغيله..."
    (ollama serve > /dev/null 2>&1 &)
    sleep 4
fi
curl -s --max-time 3 http://localhost:11434/api/tags > /dev/null || { echo "❌ تعذّر تشغيل Ollama"; exit 1; }

echo "==> 2) نموذج التضمين bge-m3"
ollama list | grep -q "bge-m3" || ollama pull bge-m3

echo "==> 3) الفهرسة"
bash run_ingest.sh

echo "==> 4) تشغيل الواجهة"
exec bash run_app.sh
