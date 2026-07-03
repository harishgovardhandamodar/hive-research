FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY hive-datatype/hive_datatype.py /app/hive_datatype/hive_datatype.py
COPY hive-datatype/__init__.py /app/hive_datatype/__init__.py

COPY hive-research/pyproject.toml /app/hive-research/pyproject.toml
COPY hive-research/hive_research/ /app/hive-research/hive_research/
COPY hive-research/config.yaml /app/config.yaml

RUN pip install --no-cache-dir /app/hive-research/
RUN cp /app/hive-research/hive_research/dashboard.html /usr/local/lib/python3.13/site-packages/hive_research/dashboard.html

RUN groupadd -r hive && useradd -r -g hive -d /app -s /bin/false hive
RUN chown -R hive:hive /app /usr/local/lib/python3.13/site-packages/
USER hive

ENV PYTHONPATH=/app
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434
ENV OLLAMA_MODEL=llama3.2:3b
ENV OLLAMA_FAST_MODEL=llama3.2:3b
ENV OLLAMA_EMBED_MODEL=nomic-embed-text

EXPOSE 7777
VOLUME ["/app/data"]

ENTRYPOINT ["python", "-m", "hive_research", "serve", "--host", "0.0.0.0", "--port", "7777"]
