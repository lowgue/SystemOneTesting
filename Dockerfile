FROM node:24-slim AS frontend
WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements/ requirements/
RUN pip install -r requirements/web.txt
COPY pyproject.toml ./
COPY vision_bench/ vision_bench/
COPY benchmarks/ benchmarks/
COPY eval/images/cars/ eval/images/cars/
COPY eval/figures/scientific_20261026_final/ eval/figures/scientific_20261026_final/
COPY --from=frontend /build/dist frontend/dist/
RUN pip install --no-deps -e . && useradd --uid 10001 --create-home appuser && mkdir -p .artifacts/studio && chown -R appuser:appuser .artifacts
ENV HOME=/home/appuser
USER appuser
EXPOSE 7860
CMD ["uvicorn", "vision_bench.webui.api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--timeout-graceful-shutdown", "5"]
