import json
from typing import List
import uuid
from abc import abstractmethod

import chromadb
import pandas as pd
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os 
from vanna.base import VannaBase
from chromadb.config import DEFAULT_DATABASE, DEFAULT_TENANT

default_ef = embedding_functions.DefaultEmbeddingFunction()


class ChromaDB_VectorStore(VannaBase):
    def __init__(self, config=None):
        VannaBase.__init__(self, config=config)

        if config is not None:
            if config.get("authtype") == 'BASIC':
                path = config.get("path", ".")
                self.embedding_function = config.get("embedding_function", default_ef)
                self.chroma_client = chromadb.PersistentClient(
                    path=path, settings=Settings(anonymized_telemetry=False)
                )
            elif config.get("authtype") == 'TOKEN':
                
                self.embedding_function = config.get("embedding_function", default_ef)
                self.chroma_client = chromadb.HttpClient( host=os.environ.get('CHROMA_SERVER_HOST'),
                                      port=os.environ.get('CHROMA_SERVER_PORT'), tenant=DEFAULT_TENANT, database=DEFAULT_DATABASE,
                                        settings=Settings(chroma_client_auth_provider= "token",
                                        chroma_client_auth_credentials=os.environ.get('CHROMA_SERVER_AUTH_CREDENTIALS'),
                                        anonymized_telemetry=False))
                                      
                                    #   )
            else:
                path = "."
                self.embedding_function = default_ef
                self.chroma_client = chromadb.PersistentClient(
                    path=path, settings=Settings(anonymized_telemetry=False)
                )

        self.documentation_collection = self.chroma_client.get_or_create_collection(
            name="documentation", embedding_function=self.embedding_function
        )
        self.ddl_collection = self.chroma_client.get_or_create_collection(
            name="ddl", embedding_function=self.embedding_function
        )
        self.sql_collection = self.chroma_client.get_or_create_collection(
            name="sql", embedding_function=self.embedding_function
        )

    def generate_embedding(self, data: str, **kwargs) -> List[float]:
        embedding = self.embedding_function([data])
        if len(embedding) == 1:
            return embedding[0]
        return embedding

    def add_question_sql(self, question: str, sql: str, schema:str =None, **kwargs) -> str:
        question_sql_json = json.dumps(
            {
                "question": question,
                "sql": sql,
                "Active": True,  # Adding the Active field with a value of True
                "SchemaName": schema if schema is not None else "PH General v1",  # Adding the SchemaName field
            }
        )
        id = str(uuid.uuid4()) + "-sql"
        self.sql_collection.add(
            documents=question_sql_json,
            embeddings=self.generate_embedding(question_sql_json),
            metadatas = [{"Active":True, "SchemaName": schema}],
            ids=id,
        )

        return id

    def add_ddl(self, ddl: str,schema:str =None, **kwargs) -> str:
        id = str(uuid.uuid4()) + "-ddl"

        self.ddl_collection.add(
            documents=ddl,
            embeddings=self.generate_embedding(ddl),
            metadatas = [{"Active":True, "SchemaName": schema}],
            ids=id,
        )
        return id

    def add_documentation(self, documentation: str,schema:str =None, **kwargs) -> str:
        id = str(uuid.uuid4()) + "-doc"
        self.documentation_collection.add(
            documents=documentation,
            embeddings=self.generate_embedding(documentation),
            metadatas = [{"Active":True, "SchemaName": schema}],
            ids=id,
        )
        return id

    def get_training_data(self,schema:str =None, **kwargs) -> pd.DataFrame:
        sql_data = self.sql_collection.get(where= {
                        "$and": [
                            {"Active": True},
                            {"SchemaName": schema}
                        ]
                    })

        df = pd.DataFrame()

        if sql_data is not None:
            # Extract the documents and ids
            documents = [json.loads(doc) for doc in sql_data["documents"]]
            ids = sql_data["ids"]

            # Create a DataFrame
            df_sql = pd.DataFrame(
                {
                    "id": ids,
                    "question": [doc["question"] for doc in documents],
                    "content": [doc["sql"] for doc in documents],
                }
            )

            df_sql["training_data_type"] = "sql"

            df = pd.concat([df, df_sql])

        ddl_data = self.ddl_collection.get(where= {
                        "$and": [
                            {"Active": True},
                            {"SchemaName": schema}
                        ]
                    })

        if ddl_data is not None:
            # Extract the documents and ids
            documents = [doc for doc in ddl_data["documents"]]
            ids = ddl_data["ids"]

            # Create a DataFrame
            df_ddl = pd.DataFrame(
                {
                    "id": ids,
                    "question": [None for doc in documents],
                    "content": [doc for doc in documents],
                }
            )

            df_ddl["training_data_type"] = "ddl"

            df = pd.concat([df, df_ddl])

        doc_data = self.documentation_collection.get(where= {
                        "$and": [
                            {"Active": True},
                            {"SchemaName": schema}
                        ]
                    })

        if doc_data is not None:
            # Extract the documents and ids
            documents = [doc for doc in doc_data["documents"]]
            ids = doc_data["ids"]

            # Create a DataFrame
            df_doc = pd.DataFrame(
                {
                    "id": ids,
                    "question": [None for doc in documents],
                    "content": [doc for doc in documents],
                }
            )

            df_doc["training_data_type"] = "documentation"

            df = pd.concat([df, df_doc])

        return df

    def remove_training_data(self, id: str, **kwargs) -> bool:
        if id.endswith("-sql"):
            self.sql_collection.delete(ids=id)
            return True
        elif id.endswith("-ddl"):
            self.ddl_collection.delete(ids=id)
            return True
        elif id.endswith("-doc"):
            self.documentation_collection.delete(ids=id)
            return True
        else:
            return False
    
    def remove_collection(self, collection_name: str) -> bool:
        """
        This function can reset the collection to empty state. 

        Args:
            collection_name (str): sql or ddl or documentation

        Returns:
            bool: True if collection is deleted, False otherwise
        """
        if collection_name == "sql":
            self.chroma_client.delete_collection(name="sql")
            self.sql_collection = self.chroma_client.get_or_create_collection(
                name="sql", embedding_function=self.embedding_function
            )
            return True
        elif collection_name == "ddl":
            self.chroma_client.delete_collection(name="ddl")
            self.ddl_collection = self.chroma_client.get_or_create_collection(
                name="ddl", embedding_function=self.embedding_function
            )
            return True
        elif collection_name == "documentation":
            self.chroma_client.delete_collection(name="documentation")
            self.documentation_collection = self.chroma_client.get_or_create_collection(
                name="documentation", embedding_function=self.embedding_function
            )
            return True
        else:
            return False
    
    @staticmethod
    def _extract_documents(query_results) -> list:
        """
        Static method to extract the documents from the results of a query.

        Args:
            query_results (pd.DataFrame): The dataframe to use.
        
        Returns:
            List[str] or None: The extracted documents, or an empty list or single document if an error occurred.
        """
        if query_results is None:
            return []

        if "documents" in query_results:
            documents = query_results["documents"]

            if len(documents) == 1 and isinstance(documents[0], list):
                try:
                    documents = [json.loads(doc) for doc in documents[0]]
                except Exception as e:
                    return documents[0]

            return documents

    def get_similar_question_sql(self, question: str,schema:str =None, **kwargs) -> list:
        return ChromaDB_VectorStore._extract_documents(
            self.sql_collection.query(
                query_texts=[question],
                where= {
                        "$and": [
                            {"Active": True},
                            {"SchemaName": schema}
                        ]
                    }
            )
        )

    def get_related_ddl(self, question: str,schema:str =None, **kwargs) -> list:
        return ChromaDB_VectorStore._extract_documents(
            self.ddl_collection.query(
                query_texts=[question],
                where= {
                        "$and": [
                            {"Active": True},
                            {"SchemaName": schema}
                        ]
                    }
            )
        )

    def get_related_documentation(self, question: str,schema:str =None, **kwargs) -> list:

        return ChromaDB_VectorStore._extract_documents(
            self.documentation_collection.query(
                query_texts=[question],
                where= {
                        "$and": [
                            {"Active": True},
                            {"SchemaName": schema}
                        ]
                    }
            )
        )