from datetime import datetime
from pymongo import MongoClient
import time
import openai
import uuid,os
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv(".env")
# gets the API Key from environment variable AZURE_OPENAI_API_KEY
AzureOpenAIclient = AzureOpenAI(
    # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
    api_version="2023-07-01-preview",
    # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
    azure_endpoint= os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZUREOPENAIKEY")
)
openai.api_type = 'azure'
openai.api_key = os.environ.get("AZUREOPENAIKEY")
openai.api_base = os.environ.get("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2023-07-01-preview"
embeddings_deployment = 'text-embedding-ada-002'

def generate_embeddings(text):
    '''
    Generate embeddings from string of text.
    This will be used to vectorize data and user input for interactions with Azure OpenAI.
    '''
    response =  AzureOpenAIclient.embeddings.create(
        input=text, model="text-embedding-ada-002")
    embeddings = response.data[0].embedding
    time.sleep(0.5) # rest period to avoid rate limiting on AOAI for free tier
    return embeddings
default_ef = generate_embeddings

#https://github.com/microsoft/AzureDataRetrievalAugmentedGenerationSamples/blob/main/Python/CosmosDB-MongoDB-vCore/CosmosDB-MongoDB-vCore_AzureOpenAI_Tutorial.ipynb

class MyCustomVectorDB:
    def __init__(self, config=None):
        if config is None:
            config = {'cosmos_db_uri': 'your_default_db_uri', 'cosmos_db_name': 'vanna_ai'}
        
        self.client = MongoClient(config['cosmos_db_uri'])

        # Check if the database exists
        if config['cosmos_db_name'] not in self.client.list_database_names():
            print(f"Database '{config['cosmos_db_name']}' does not exist. It will be created when the first collection is created.")
        self.db = self.client[config['cosmos_db_name']]
        self.embedding_function = config.get("embedding_function", default_ef)


        # Define the collections required by your application
        self.required_collections = ['ddl', 'documentation', 'question_sql']
        
        # Ensure these collections exist
        for collection_name in self.required_collections:
            self.create_collection(collection_name, create_vector_index=True)


    def create_collection(self, collection_name: str, create_vector_index: bool = True):        
        if collection_name not in self.db.list_collection_names():
            self.db.create_collection(collection_name)
            print(f"Created collection '{collection_name}'.\n")
        
        if create_vector_index:
            # Assuming 'contentVector' is the field you want to index. Adjust accordingly.
            self.db.command({
                'createIndexes': collection_name,
                'indexes': [
                    {
                        'name': 'VectorSearchIndex',
                        'key': {"contentVector": "cosmosSearch"},
                        'cosmosSearchOptions': {
                            'kind': 'vector-ivf',
                            'numLists': 1,
                            'similarity': 'COS',
                            'dimensions': 1536
                        }
                    }
                ]
            })
            print(f"Created vector search index on '{collection_name}'.\n")
        else:
            print(f"Using collection: '{collection_name}'.\n")

    def add_question_sql(self, question: str, sql: str, schema: str = None, active: bool = True, insert_date: datetime = None, inserted_by: str = None, **kwargs) -> str:
        question_vector = generate_embeddings(question)
        sql_vector = generate_embeddings(sql)
        document = {
            'id': str(uuid.uuid4()) + "-sql",
            'question': question,
            'sql': sql,
            'question_vector': question_vector,
            'sql_vector': sql_vector,
            'schema': schema,
            'active': active,
            'insert_date': insert_date if insert_date else datetime.now(),
            'inserted_by': inserted_by
        }
        result = self.db.question_sql.insert_one(document)
        return str(result.inserted_id)
   
    def add_ddl(self, ddl: str, schema: str = None, active: bool = True, insert_date: datetime = None, inserted_by: str = None, **kwargs) -> str:
        vector_embedding = generate_embeddings(ddl)
        document = {
            'id':str(uuid.uuid4()) + "-ddl",
            'ddl': ddl,
            'vector': vector_embedding,
            'schema': schema,
            'active': active,
            'insert_date': insert_date if insert_date else datetime.now(),
            'inserted_by': inserted_by
        }
        result = self.db.ddl.insert_one(document)
        return str(result.inserted_id)
    
    def add_documentation(self, doc: str, schema: str = None, active: bool = True, insert_date: datetime = None, inserted_by: str = None, **kwargs) -> str:
        vector_embedding = generate_embeddings(doc)
        document = {
            'id': str(uuid.uuid4()) + "-doc",
            'doc': doc,
            'vector': vector_embedding,
            'schema': schema,
            'active': active,
            'insert_date': insert_date if insert_date else datetime.now(),
            'inserted_by': inserted_by
        }
        result = self.db.documentation.insert_one(document)
        return str(result.inserted_id)
   
    def get_related_documentation(self, question: str, schema=None, **kwargs) -> list:
        question_vector = generate_embeddings(question)
        related_docs = self._vector_search(self.db.documentation, question_vector, "contentVector", schema=schema)
        return [doc for doc in related_docs]

    def get_similar_question_sql(self, question: str, schema=None, **kwargs) -> list:
      question_vector = generate_embeddings(question)
      similar_questions = self._vector_search(self.db.question_sql, question_vector, "contentVector", schema=schema)
      return [question for question in similar_questions]

    def get_related_ddl(self, question: str, schema=None, **kwargs) -> list:
        """
        Retrieves related DDLs based on a query question.
        
        Args:
            question (str): The question based on which related DDLs are to be found.
            schema (str, optional): A schema to filter the DDLs. Defaults to None.
        
        Returns:
            list: A list of related DDL documents.
        """
        question_vector = generate_embeddings(question)
        related_ddls = self._vector_search(self.db.ddl, question_vector, "contentVector", schema=schema)
        return [ddl['document'] for ddl in related_ddls]


    def _vector_search(self, collection, vector, field='vector', schema=None, num_results=3):
        pipeline = [
            {'$match': {'active': True}},  # Filter for active documents
            {'$search': {
                'cosmosSearch': {
                    'vector': vector,
                    'path': field,
                    'k': num_results  #, "efsearch": 40 # optional for HNSW only
                },
                'returnStoredSource': True
            }},
            {'$project': {'similarityScore': {'$meta': 'searchScore'}, 'document': '$$ROOT'}}
        ]

        # If a schema is specified, add a match stage to filter by schema
        if schema:
            pipeline.insert(0, {'$match': {'schema': schema}})

        results = collection.aggregate(pipeline)
        return list(results)  # Convert cursor to list before returning