from datetime import datetime
from pymongo import MongoClient
import time
import openai
import uuid,os
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import pandas as pd

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

class AzureCosmosMongovCoreDB:
    def __init__(self, config=None):
        if config is None:
            config = {'cosmos_mongo_user':'??','cosmos_mongo_pass':'??','cosmos_mongo_server': 'your_default_db_uri', 'cosmos_mongo_db_name': 'vanna_ai'}
            
        self.client = MongoClient("mongodb+srv://"+config['cosmos_mongo_user']+":"+config['cosmos_mongo_pass']+"@"+config['cosmos_mongo_server']+"?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000")
        # Check if the database exists
        if config['cosmos_mongo_db_name'] not in self.client.list_database_names():
            print(f"Database '{config['cosmos_mongo_db_name']}' does not exist. It will be created when the first collection is created.")
        self.db = self.client[config['cosmos_mongo_db_name']]
        self.embedding_function = config.get("embedding_function", default_ef)


        # Define the collections required by your application
        self.required_collections = {'ddl_collection':'contentVector', 'documentation_collection':'contentVector', 'sql_collection':'question_vector'}
        
        self.client.drop_database(config['cosmos_mongo_db_name'])
        self.db.ddl_collection.drop_indexes()
        self.db.documentation_collection.drop_indexes()
        self.db.sql_collection.drop_indexes()

        # Ensure these collections exist
        for collection_name in self.required_collections.keys():
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
                            'key': {self.required_collections[collection_name]: "cosmosSearch"},
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
        result = self.db.sql_collection.insert_one(document)
        return str(result.inserted_id)
   
    def add_ddl(self, ddl: str, schema: str = None, active: bool = True, insert_date: datetime = None, inserted_by: str = None, **kwargs) -> str:
        vector_embedding = generate_embeddings(ddl)
        document = {
            'id':str(uuid.uuid4()) + "-ddl",
            'ddl': ddl,
            'contentVector': vector_embedding,
            'schema': schema,
            'active': active,
            'insert_date': insert_date if insert_date else datetime.now(),
            'inserted_by': inserted_by
        }
        result = self.db.ddl_collection.insert_one(document)
        return str(result.inserted_id)
    
    def add_documentation(self, doc: str, schema: str = None, active: bool = True, insert_date: datetime = None, inserted_by: str = None, **kwargs) -> str:
        vector_embedding = generate_embeddings(doc)
        document = {
            'id': str(uuid.uuid4()) + "-doc",
            'doc': doc,
            'contentVector': vector_embedding,
            'schema': schema,
            'active': active,
            'insert_date': insert_date if insert_date else datetime.now(),
            'inserted_by': inserted_by
        }
        result = self.db.documentation_collection.insert_one(document)
        return str(result.inserted_id)
   
    def get_related_documentation(self, question: str, schema=None,cutoff:float=0.75, **kwargs) -> list:
        question_vector = generate_embeddings(question)
        related_docs = self._vector_search(self.db.documentation_collection, question_vector, self.required_collections['documentation_collection'], schema=schema)
        return self._extract_documents([doc for doc in related_docs if doc['similarityScore'] >= cutoff], schema)

    def get_similar_question_sql(self, question: str, schema=None,cutoff:float=0.75, **kwargs) -> list:
        question_vector = generate_embeddings(question)
        similar_questions = self._vector_search(self.db.sql_collection, question_vector, self.required_collections['sql_collection'], schema=schema)
        return self._extract_documents([question for question in similar_questions if question['similarityScore']],schema)

    def get_related_ddl(self, question: str, schema=None,cutoff:float=0.75, **kwargs) -> list:
        """
        Retrieves related DDLs based on a query question.
        
        Args:
            question (str): The question based on which related DDLs are to be found.
            schema (str, optional): A schema to filter the DDLs. Defaults to None.
        
        Returns:
            list: A list of related DDL documents.
        """
        question_vector = generate_embeddings(question)
        related_ddls = self._vector_search(self.db.ddl_collection, question_vector, self.required_collections['ddl_collection'], schema=schema)
        return self._extract_documents([ddl for ddl in related_ddls if ddl['similarityScore'] >= cutoff],schema)
    
    def _vector_search(self, collection, vectory_query, field='vector', schema=None, num_results=3):

        pipeline = [
            {
                '$search': {
                    "cosmosSearch": {
                        "vector": vectory_query,
                        "path": field,
                        "k": num_results, #, "efsearch": 40 # optional for HNSW only 
                        
                    },
                    "returnStoredSource": True }},
                {'$project': {'similarityScore': { '$meta': 'searchScore' }, 'document' : '$$ROOT' } }     
            
        ]
        results = collection.aggregate(pipeline)
        return results
    
    @staticmethod
    def _extract_documents(query_results,schema) -> list:
        """
        Static method to extract the documents from the results of a query.

        Args:
            query_results (pd.DataFrame): The dataframe to use.
        
        Returns:
            List[str] or None: The extracted documents, or an empty list or single document if an error occurred.
        """
        documents=[]
        if len(query_results)<0:
            return []
        try:
            if "doc" in query_results[0]['document']:
                for result in query_results:
                        if result['document']['schema'] == schema and result['document']['active'] == True:
                            documents.append(result['document']["doc"])

            elif "ddl" in query_results[0]['document']:
                for result in query_results:
                    if result['document']['schema'] == schema and result['document']['active'] == True:
                        documents.append(result['document']["ddl"])
        
            elif "question" in query_results[0]['document']:
                for result in query_results:
                    if result['document']['schema'] == schema and result['document']['active'] == True:
                        documents.append([result['document']["question"],result['document']["sql"]])
            else:
                Exception('Error -  no expected document type returned from vector search')
        
        except Exception as e:
                    print(query_results)
        

        return documents
    
    def remove_training_data(self, id: str, **kwargs) -> bool:
        if id.endswith("-sql"):
            result = self.sql_collection.delete_one({'_id': id})
        elif id.endswith("-ddl"):
            result = self.ddl_collection.delete_one({'_id': id})
        elif id.endswith("-doc"):
            result = self.documentation_collection.delete_one({'_id': id})
        else:
            return False

        # Check if a document was actually deleted
        return result.deleted_count > 0

    def get_training_data(self, schema: str = None, **kwargs) -> pd.DataFrame:
        query_filter = {"active": True}
        if schema:
            query_filter["schema"] = schema
        
        # Initialize an empty DataFrame
        df = pd.DataFrame()
        
        # Fetch and process SQL data
        sql_data = self.db.sql_collection.find(query_filter)
        df_sql = pd.DataFrame(list(sql_data))
        if not df_sql.empty:
            df_sql = df_sql.rename(columns={'question': 'question', 'sql': 'content'})
            df_sql['training_data_type'] = 'sql'
            df = pd.concat([df, df_sql[['id', 'question', 'content', 'training_data_type']]], ignore_index=True)
        
        # Fetch and process DDL data
        ddl_data = self.db.ddl_collection.find(query_filter)
        df_ddl = pd.DataFrame(list(ddl_data))
        if not df_ddl.empty:
            df_ddl['question'] = None  # Assuming there's no 'question' for DDL entries
            df_ddl = df_ddl.rename(columns={'ddl': 'content'})
            df_ddl['training_data_type'] = 'ddl'
            df = pd.concat([df, df_ddl[['id', 'question', 'content', 'training_data_type']]], ignore_index=True)
        
        # Fetch and process Documentation data
        doc_data = self.db.documentation_collection.find(query_filter)
        df_doc = pd.DataFrame(list(doc_data))
        if not df_doc.empty:
            df_doc['question'] = None  # Assuming there's no 'question' for Documentation entries
            df_doc = df_doc.rename(columns={'doc': 'content'})
            df_doc['training_data_type'] = 'documentation'
            df = pd.concat([df, df_doc[['id', 'question', 'content', 'training_data_type']]], ignore_index=True)
        
        return df