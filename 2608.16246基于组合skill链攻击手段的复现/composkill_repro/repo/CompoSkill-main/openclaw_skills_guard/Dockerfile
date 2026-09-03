FROM python:3.11-slim

LABEL maintainer="AicoGuardClaw" \
      description="AI Skill Security Audit Service"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HOST=0.0.0.0 \
    PORT=7860 \
    MAX_UPLOAD_SIZE_MB=50 \
    MAX_CONTENT_CHARS=500000

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["python", "main.py"]
