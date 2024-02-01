FROM python:3.9

RUN pip install poetry==1.4.2

#WORKDIR /

# COPY /DataGenieApp/pyproject.toml /DataGenieApp/poetry.lock ./
COPY DataGenieApp ./DataGenieApp
WORKDIR /DataGenieApp
RUN touch README.md

RUN poetry install --without dev

ENTRYPOINT ["poetry", "run", "streamlit","run", "dataGenieApp.py"]