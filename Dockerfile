
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04


WORKDIR /app


RUN apt-get update && apt-get install -y \
    software-properties-common curl git wget \
    build-essential libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender1 zlib1g-dev libffi-dev \
    libssl-dev libbz2-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libreadline-dev libsqlite3-dev liblzma-dev ca-certificates \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*


RUN wget https://www.python.org/ftp/python/3.12.2/Python-3.12.2.tgz && \
    tar -xf Python-3.12.2.tgz && cd Python-3.12.2 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && make altinstall && \
    cd .. && rm -rf Python-3.12.2*


RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12


RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.12 2 && \
    update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3.12 2


RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128


RUN pip3 install git+https://github.com/facebookresearch/segment-anything.git


RUN pip3 install git+https://github.com/openai/CLIP.git


COPY requirements.txt .
RUN pip3 install -r requirements.txt


RUN python3.12 -m spacy download en_core_web_sm

COPY . .

RUN apt-get update && apt-get install -y wget


