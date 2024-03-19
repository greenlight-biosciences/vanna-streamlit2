from datetime import datetime
from pymongo import MongoClient
import time
import openai
import uuid,os
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import pandas as pd
from vanna.base import VannaBase
from azureopenai_embedding import AzureOpenAI_Embeddings as aoi_e
from typing import List
from VannaLogger import VannaLogger

#https://github.com/microsoft/AzureDataRetrievalAugmentedGenerationSamples/blob/main/Python/CosmosDB-MongoDB-vCore/CosmosDB-MongoDB-vCore_AzureOpenAI_Tutorial.ipynb

class AzureCosmosMongovCoreDB(VannaBase):
    def __init__(self, config=None):
        VannaBase.__init__(self, config=config)
        self.aoi_e = aoi_e(config=config)
        self.vannaLogger = VannaLogger()
        if config is None:
            config = {'cosmos_mongo_user':'??','cosmos_mongo_pass':'??','cosmos_mongo_server': 'your_default_db_uri', 'cosmos_mongo_db_name': 'vanna_ai'}
            Exception('Please update config with appropriate inputs')
        else:
            self.client = MongoClient("mongodb+srv://"+config['cosmos_mongo_user']+":"+config['cosmos_mongo_pass']+"@"+config['cosmos_mongo_server']+"?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000")
            # Check if the database exists
            if config['cosmos_mongo_db_name'] not in self.client.list_database_names():
                self.vannaLogger.logWarning(f"Database '{config['cosmos_mongo_db_name']}' does not exist. It will be created when the first collection is created.")
            self.db = self.client[config['cosmos_mongo_db_name']]
            self.generate_embedding_func = self.aoi_e.generate_embedding
            self.cutoff = float(config.get("cutoff", 0.75))
            self.AzureOpenAIclient = AzureOpenAI(
            api_version= config["api_version"],
            azure_endpoint= config["api_base"],
            api_key=config["api_key"])
            

        # Define the collections required by your application
        self.required_collections = {'ddl_collection':'contentVector', 'documentation_collection':'contentVector', 'sql_collection':'question_vector'}
        
        # self.client.drop_database(config['cosmos_mongo_db_name'])
        # self.db.ddl_collection.drop_indexes()
        # self.db.documentation_collection.drop_indexes()
        # self.db.sql_collection.drop_indexes()

        # Ensure these collections exist
        for collection_name in self.required_collections.keys():
            self.create_collection(collection_name, create_vector_index=True)

    def set_embedding_fuc(self, embeddingFunc):
        self.generate_embedding_func = embeddingFunc

    def generate_embedding(self, data: str, **kwargs) -> List[float]:
        self.vannaLogger.logDebug('Generating Embedding for: {data}')
        embedding = self.generate_embedding_func([data])
        if len(embedding) == 1:
            return embedding[0]
        return embedding
    
    def update_cutoff(self, newcutoff:float=0.75):
        self.cutoff = newcutoff
        self.vannaLogger.logInfo(f'Update similarity cutoff to: {self.cutoff}')


    def create_collection(self, collection_name: str, create_vector_index: bool = True):        
        if collection_name not in self.db.list_collection_names():
            self.db.create_collection(collection_name)
            self.vannaLogger.logInfo(f"Created collection '{collection_name}'.")
        
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

                self.vannaLogger.logInfo(f"Created vector search index on '{collection_name}'.")
        else:
            self.vannaLogger.logInfo(f"Using collection: '{collection_name}'.")

    def add_question_sql(self, question: str, sql: str, schema: str = None, active: bool = True, insert_date: datetime = None, inserted_by: str = None, **kwargs) -> str:
        question_vector = self.generate_embedding(question)
        sql_vector = self.generate_embedding(sql)
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
        vector_embedding = self.generate_embedding(ddl)
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
        vector_embedding = self.generate_embedding(doc)
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
   
    def get_related_documentation(self,question: str=None, questionVectorIN=None, schema=None,cutoff:float=0.75, **kwargs) -> list:
        if questionVectorIN is  None and question is not None:
            question_vector = self.generate_embedding(question)
        elif questionVectorIN is not  None and question is None:
             question_vector = questionVectorIN
        else:
            Exception('Submit Either question or questionVectorIN into get_related_documentation function')
        related_docs = self._vector_search(collection=self.db.documentation_collection, vectory_query=question_vector,field= self.required_collections['documentation_collection'], schema=schema, cutoff=self.cutoff)
        # return self._extract_documents([doc for doc in related_docs if doc['similarityScore'] >= cutoff], schema)
        return self.extract_documents(query_results=[doc for doc in related_docs], datatype='documentation')

    def get_similar_question_sql(self, question: str=None, questionVectorIN=None, schema=None,cutoff:float=0.75, **kwargs) -> list:
        if questionVectorIN is  None and question is not None:
            question_vector = self.generate_embedding(question)
        elif questionVectorIN is not  None and question is None:
             question_vector = questionVectorIN
        else:
            Exception('Submit Either question or questionVectorIN into get_similar_question_sql function')
        similar_questions = self._vector_search(collection=self.db.sql_collection, vectory_query= question_vector, field=self.required_collections['sql_collection'], schema=schema, cutoff=self.cutoff)
        # return self._extract_documents([question for question in similar_questions if question['similarityScore']],schema)
        return self.extract_documents(query_results=[question for question in similar_questions], datatype='sql')

    def get_related_ddl(self, question: str=None, questionVectorIN=None, schema=None,cutoff:float=0.75, **kwargs) -> list:
        """
        Retrieves related DDLs based on a query question.
        
        Args:
            question (str): The question based on which related DDLs are to be found.
            schema (str, optional): A schema to filter the DDLs. Defaults to None.
        
        Returns:
            list: A list of related DDL documents.
        """
        if questionVectorIN is  None and question is not None:
            question_vector = self.generate_embedding(question)
        elif questionVectorIN is not  None and question is None:
             question_vector = questionVectorIN
        else:
            Exception('Submit Either question or questionVectorIN into get_related_ddl function')

        related_ddls = self._vector_search(collection=self.db.ddl_collection, vectory_query=question_vector, field=self.required_collections['ddl_collection'], schema=schema, cutoff=self.cutoff)
        # return self.extract_documents([ddl for ddl in related_ddls if ddl['similarityScore'] >= cutoff],schema)
        return self.extract_documents(query_results=[ddl for ddl in related_ddls], datatype='ddl')

    def _vector_search(self, collection, vectory_query, field='vector', schema=None,cutoff:float =0.75, num_results=10):
        self.vannaLogger.logDebug(f'Running vector Search for : {vectory_query}')
        pipeline = [
            {
                '$search': {
                    "cosmosSearch": {
                        "vector": vectory_query,
                        "path": field,
                        "k": num_results, #, "efsearch": 40 # optional for HNSW only 
                        
                    },
                    "returnStoredSource": True }},
                {'$project': {'similarityScore': { '$meta': 'searchScore' }, 'document' : '$$ROOT' } }  ,
                {'$match': {'similarityScore': {'$gte': self.cutoff}}},  # Filter by similarity score >= 0.75
                {'$match': {'document.schema': schema, 'document.active': True}}  # Further filter on `document`  
            
        ]
        results = collection.aggregate(pipeline)
        return results
    
    def extract_documents(self,query_results, datatype:str=None, schema:str=None) -> list:
        """
        Static method to extract the documents from the results of a query.

        Args:
            query_results (pd.DataFrame): The dataframe to use.
        
        Returns:
            List[str] or None: The extracted documents, or an empty list or single document if an error occurred.
        """
        documents=[]
        if len(query_results)==0:
            self.vannaLogger.logInfo(f'No Results for: {datatype}')
            return []
        try:
            if "doc" in query_results[0]['document']:
                for result in query_results:
                    # if result['document']['schema'] == schema and result['document']['active'] == True:
                    documents.append(result['document']["doc"])

            elif "ddl" in query_results[0]['document']:
                for result in query_results:
                    # if result['document']['schema'] == schema and result['document']['active'] == True:
                    documents.append(result['document']["ddl"])
        
            elif "question" in query_results[0]['document']:
                for result in query_results:
                    # if result['document']['schema'] == schema and result['document']['active'] == True:
                    documents.append({'question':result['document']["question"],'sql':result['document']["sql"]})
            else:
                Exception('Error -  no expected document type returned from vector search')
        
        except Exception as e:
            self.vannaLogger.logError(f'Error: {e} | query_results:{query_results}')
  

        return documents
    
    def remove_training_data(self, id: str, **kwargs) -> bool:
        self.vannaLogger.logInfo(f'Removing Training Data for: {id}')
        if id.endswith("-sql"):
            result = self.db.sql_collection.delete_one({'id': id})
        elif id.endswith("-ddl"):
            result = self.db.ddl_collection.delete_one({'id': id})
        elif id.endswith("-doc"):
            result = self.db.documentation_collection.delete_one({'id': id})
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