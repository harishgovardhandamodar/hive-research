FROM python:3.13-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY hive-datatype/ /app/hive-datatype/
RUN pip install --no-cache-dir /app/hive-datatype/

COPY hive-research/ /app/hive-research/
RUN pip install --no-cache-dir -e /app/hive-research/

FROM python:3.13-slim

RUN groupadd -r hive && useradd -r -g hive -d /app -s /bin/false hive
RUN mkdir -p /app/data /app/hive-research && chown -R hive:hive /app

COPY --from=builder /usr/local/lib/python3.13/site-packages/ /usr/local/lib/python3.13/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/
COPY --from=builder /app/hive-research/hive_research /app/hive_research
COPY --from=builder /app/hive-research/config.yaml /app/config.yaml

WORKDIR /app
USER hive

EXPOSE 7777

ENV OLLAMA_BASE_URL=http://host.docker.internal:11434
ENV OLLAMA_MODEL=llama3.2:3b
ENV OLLAMA_FAST_MODEL=llama3.2:3b
ENV OLLAMA_EMBED_MODEL=nomic-embed-text

VOLUME ["/app/data"]

ENTRYPOINT ["python", "-m", "hive_research", "serve", "--host", "0.0.0.0", "--port", "7777"]
