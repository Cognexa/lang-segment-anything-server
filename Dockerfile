FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    openssh-client \
    build-essential \
    git

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY lang_sam lang_sam
COPY server.py server.py

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8082", "--reload"]
