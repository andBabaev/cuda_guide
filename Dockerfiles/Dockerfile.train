FROM nvidia/cuda:11.2.2-base

WORKDIR /app

RUN apt update && apt install python3-pip -y && \
    rm -rf /var/lib/apt/lists/* && \
    yes Y | apt-get purge   --auto-remove && \
    apt-get clean

COPY requirements-app.txt .
RUN pip install -r requirements-app.txt --no-cache-dir

COPY models/ models/
COPY app.py .
COPY cnn.py .
COPY config.py .
COPY utils.py .
COPY data/ data/

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
