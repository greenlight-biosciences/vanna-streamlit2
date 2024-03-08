FROM python:3.9

RUN pip install poetry==1.4.2

# Poetry's configuration:
ENV  POETRY_NO_INTERACTION=1 \
  POETRY_VIRTUALENVS_CREATE=false \
  POETRY_CACHE_DIR='/var/cache/pypoetry' \
  POETRY_HOME='/usr/local'

WORKDIR /usr/src/app

COPY /pyproject.toml /poetry.lock /README.md ./

# Install project dependencies
RUN poetry config virtualenvs.create false
RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR

# Set environment variables
ENV MODEL_NAME all-MiniLM-L6-v2
ENV DOWNLOAD_PATH /root/.cache/chroma/onnx_models/${MODEL_NAME}/onnx
ENV EXTRACTED_FOLDER_NAME onnx
ENV ARCHIVE_FILENAME onnx.tar.gz
ENV MODEL_DOWNLOAD_URL https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz
ENV MODEL_SHA256 913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3

# Download and extract the model
RUN mkdir -p ${DOWNLOAD_PATH} && \
    wget -O ${DOWNLOAD_PATH}/${ARCHIVE_FILENAME} ${MODEL_DOWNLOAD_URL} && \
    echo "${MODEL_SHA256} ${DOWNLOAD_PATH}/${ARCHIVE_FILENAME}" | sha256sum -c && \
    tar -xzf ${DOWNLOAD_PATH}/${ARCHIVE_FILENAME} -C ${DOWNLOAD_PATH} && \
    mv ${DOWNLOAD_PATH}/${EXTRACTED_FOLDER_NAME}/* ${DOWNLOAD_PATH}/ && \
    rm -rf ${DOWNLOAD_PATH}/${EXTRACTED_FOLDER_NAME} ${DOWNLOAD_PATH}/${ARCHIVE_FILENAME}

COPY DataGenie .

WORKDIR /usr/src/app
EXPOSE 80

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["poetry", "run", "streamlit","run", "dataGenieApp.py" , "--server.port=80", "--server.address=0.0.0.0"]