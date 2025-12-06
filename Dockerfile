FROM ubuntu:20.04

# Установка переменных окружения
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    python3-opencv \
    ca-certificates \
    python3-dev \
    git \
    wget \
    sudo \
    ninja-build \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Создание символической ссылки python -> python3
RUN ln -sv /usr/bin/python3 /usr/bin/python

# Создание пользователя (как в оригинальном Dockerfile)
ARG USER_ID=1000
RUN useradd -m --no-log-init --system --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

# Установка pip для пользователя
RUN wget https://bootstrap.pypa.io/pip/3.6/get-pip.py && \
    python3 get-pip.py --user && \
    rm get-pip.py

# Обновление PATH для использования локальных пакетов пользователя
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Установка зависимостей через pip --user (как в оригинальном Dockerfile)
RUN pip install --user --upgrade pip

# Установка PyTorch (CPU версия) через --user
RUN pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Установка дополнительных зависимостей через --user
RUN pip install --user opencv-python-headless pillow numpy

# Установка fvcore через --user (как в оригинальном Dockerfile)
RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'

# Клонирование Detectron2
RUN git clone https://github.com/facebookresearch/detectron2.git detectron2_repo

# Установка Detectron2 из исходников через --user
WORKDIR /home/appuser/detectron2_repo
RUN pip install --user -e .

# Клонирование репозитория cpu_version для получения inference.py и model_final.pth
RUN git clone https://github.com/PanVOUniversity/cpu_version.git /tmp/cpu_version

# Создание папки validation в detectron2
RUN mkdir -p /home/appuser/detectron2_repo/validation

# Копирование inference.py и model_final.pth в папку validation
# Проверяем оба возможных расположения файлов (в корне или в подпапке cpu-version)
RUN if [ -f /tmp/cpu_version/inference.py ]; then \
        cp /tmp/cpu_version/inference.py /home/appuser/detectron2_repo/validation/; \
    elif [ -f /tmp/cpu_version/cpu-version/inference.py ]; then \
        cp /tmp/cpu_version/cpu-version/inference.py /home/appuser/detectron2_repo/validation/; \
    else \
        echo "ERROR: inference.py not found in cpu_version repository"; exit 1; \
    fi && \
    if [ -f /tmp/cpu_version/model_final.pth ]; then \
        cp /tmp/cpu_version/model_final.pth /home/appuser/detectron2_repo/validation/; \
    elif [ -f /tmp/cpu_version/cpu-version/model_final.pth ]; then \
        cp /tmp/cpu_version/cpu-version/model_final.pth /home/appuser/detectron2_repo/validation/; \
    else \
        echo "ERROR: model_final.pth not found in cpu_version repository"; exit 1; \
    fi && \
    ls -lh /home/appuser/detectron2_repo/validation/

# Установка кэша для моделей
ENV FVCORE_CACHE="/tmp"

# Установка рабочей директории
WORKDIR /home/appuser/detectron2_repo/validation

# Очистка временных файлов
RUN rm -rf /tmp/cpu_version

# По умолчанию запускаем bash
CMD ["/bin/bash"]

