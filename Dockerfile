FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*


RUN useradd -m -u 1000 user
USER user

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -e .

EXPOSE 7860

CMD ["streamlit", "run", "application.py", "--server.port=7860", "--server.address=0.0.0.0"]

