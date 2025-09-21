FROM gcr.io/kaggle-gpu-images/python:latest

RUN pip install --no-cache-dir jupyterlab

ARG USERNAME=kaggle
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && mkdir -p /workspace \
    && chown $USERNAME:$USERNAME /workspace

USER $USERNAME
WORKDIR /workspace

EXPOSE 8888

# Запускаем Jupyter при старте контейнера
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=", "--NotebookApp.password="]
