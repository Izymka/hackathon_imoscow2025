FROM gcr.io/kaggle-gpu-images/python:v161

RUN pip install --no-cache-dir jupyterlab

ARG USERNAME=kaggle
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && mkdir -p /workspace \
    && chown $USERNAME:$USERNAME /workspace

# Исправляем права доступа для kaggle логов
RUN touch /tmp/kaggle.log && chmod 666 /tmp/kaggle.log
RUN sudo chmod 777 /var/log/pip.log


#USER $USERNAME
WORKDIR /workspace

EXPOSE 8888

# Запускаем Jupyter при старте контейнера
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=", "--NotebookApp.password="]
