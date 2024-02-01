# Vanna.AI Streamlit App
<img width="1392" alt="Screenshot 2023-06-23 at 3 49 45 PM" src="./assets/vanna_demo.gif">

## Installation

This project uses Poetry, so the first thing is to install it. 

```bash
pip install poetry
```

Poetry allows to

1. Install and manage the dependencies of the project
2. Create a clean virtual environment that is fully isolated from your current Python environment


Packages are listed in the `pyproject.toml` file. 

To install them, simply run:

```bash
poetry install --with dev
```

## Usage

If you're running the app locally, please add a `.env` file at the root of the project with your crendentials:

```bash
AZURE_OPENAI_ENDPOINT=https://XXXXX.openai.azure.com #No quotes 
AZUREOPENAIENGINE=gpt-XXX #No quotes 
AZUREOPENAIKEY=XXXXXX #No quotes 
DATABASETYPE=Snowflake #No quotes 
USER=XXXXXX #No quotes 
PASS=XXXXX #No quotes if pass requires quotes due to special char change out char
ACCOUNT=snowflake.account.name #No quotes 
WAREHOUSE=XXX #No quotes 
DATABASE=XXXX #No quotes 
SCHEMA=XXX #No quotes 
ROLE=XXX #No quotes 
APPTITLE=YOURAPPNAMEHERE #No quotes 
GETHELPURL=YOURHELPDOCSURLHERE #No quotes 
SUBMITTICKETURL=YOURTICKETINGPORTALHERE #No quotes 
```

To create a Vanna API key, please refer to this [link](https://vanna.ai/).

## Running Locally
Run the locally app with this command:

```bash
poetry run streamlit run app.py
```

## Running App with Docker
From the top directory:

1. First build your image 
```bash
docker build -t datagenie .
```

2. Run container
```bash
docker run --env-file .env -p 8501:8501 datagenie
```

## App URL
App will be accessible at: 
[http://localhost:8501/](http://localhost:8501/)

### TroubleShooting
> **_NOTE:_** IF snowflake access is failing double check that you did not quote any of the ENVs in your env file. Quoted envs passed in through docker run end up with additional quotes.



## License
[MIT](https://choosealicense.com/licenses/mit/)
