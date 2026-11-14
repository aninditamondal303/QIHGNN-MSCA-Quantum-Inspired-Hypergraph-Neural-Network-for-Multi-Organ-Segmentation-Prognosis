# One image that serves the website and the API (and runs the network when a checkpoint is in models/).
#   docker build -t qihgnn-msca .
#   docker run -p 7860:7860 qihgnn-msca        ->  http://localhost:7860
# Works unchanged on Hugging Face Spaces (Docker SDK, port 7860), Render, Railway and Fly.io (they set $PORT).

# ---------------------------------------------------------------- website
FROM node:22-slim AS web
WORKDIR /src/web
COPY web/package.json web/package-lock.json* ./
RUN if [ -f package-lock.json ]; then npm ci --no-audit --no-fund; else npm install --no-audit --no-fund; fi
COPY web/ ./
RUN npm run build

# ---------------------------------------------------------------- API + model
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=7860
WORKDIR /app
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install "numpy>=1.26,<3" "scipy>=1.11" "pillow>=10.2" "starlette>=0.37" "pydantic>=2.6" "uvicorn[standard]>=0.29"
COPY ml/ ml/
COPY api/ api/
COPY results/ results/
COPY models/ models/
COPY --from=web /src/web/dist web/dist
COPY --from=web /src/web/public/data web/public/data
RUN useradd -m -u 1000 app && chown -R app /app
USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/api/health')" || exit 1
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
