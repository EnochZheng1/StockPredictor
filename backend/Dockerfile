FROM python:3.11-slim

WORKDIR /app

# Install TA-Lib C library
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget && \
    wget -q https://github.com/TA-Lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz && \
    tar xzf ta-lib-0.6.4-src.tar.gz && \
    cd ta-lib-0.6.4 && ./configure --prefix=/usr && make && make install && \
    cd .. && rm -rf ta-lib-0.6.4* && \
    apt-get purge -y build-essential wget && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

EXPOSE 4289

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "4289"]
