FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get clean


RUN pip install fastembed==0.5.1
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_dir='/app/.data/fastembed_cache')"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY /src /app/src

EXPOSE 8001

CMD ["fastapi", "run", "src/main.py", "--host", "0.0.0.0", "--port", "8001"]
