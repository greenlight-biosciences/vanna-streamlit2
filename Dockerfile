FROM python:3.9

RUN pip install poetry==1.4.2

# Poetry's configuration:
# ENV  POETRY_NO_INTERACTION=1 \
#   POETRY_VIRTUALENVS_CREATE=false \
#   POETRY_CACHE_DIR='/var/cache/pypoetry' \
#   POETRY_HOME='/usr/local'

WORKDIR /usr/src/app

COPY /pyproject.toml /poetry.lock /README.md ./
COPY DataGenie ./DataGenie

# Install project dependencies
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

WORKDIR /usr/src/app/DataGenie
EXPOSE 80

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["poetry", "run", "streamlit","run", "dataGenieApp.py" , "--server.port=80", "--server.address=0.0.0.0"]