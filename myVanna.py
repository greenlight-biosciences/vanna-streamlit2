
from typing import Union
from vanna import get_models, set_model
from vanna.openai.openai_chat import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
import snowflake.connector
import pandas as pd
from vanna.exceptions import DependencyError, ImproperlyConfigured, ValidationError
import os
import re

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)
        #VannaBase.__init__(self, config=config)

    # def get_models():
    #     return get_models()
    
    # def set_model():
    #     return set_model()
    def log(self, message: str):
        print(message)

    def generate_plotly_code(
        self, question: str = None, sql: str = None, df_metadata: str = None,  chart_instructions: Union[str, None] = None, plottingLib: str = 'Plotly', **kwargs
    ) -> str:
        if question is not None:
            if chart_instructions is not None:
                question = ( question   + " -- When plotting, follow these instructions: "   + chart_instructions  )
            else:
                question = "When plotting, follow these instructions: " + chart_instructions
            system_msg = f"The following is a pandas DataFrame that contains the results of the query that answers the question the user asked: '{question}'"
        else:
            system_msg = "The following is a pandas DataFrame "

        if sql is not None:
            system_msg += f"\n\nThe DataFrame was produced using this query: {sql}\n\n"

        system_msg += f"The following is information about the resulting pandas DataFrame 'df': \n{df_metadata}"

        message_log = [
            self.system_message(system_msg),
            self.user_message(
               f"Can you generate the Python {plottingLib} code to chart the results of the dataframe? Assume the data is in a pandas dataframe called 'df'. If there is only one value in the dataframe, use an Indicator. Respond with only Python {plottingLib} code. Do not answer with any explanations -- just the code."
            ),
        ]

        plotly_code = self.submit_prompt(message_log, kwargs=kwargs)
        return self._sanitize_plotly_code(self._extract_python_code(plotly_code))
    
    def edit_plotly_code(
        self, question: str = None, sql: str = None, df_metadata: str = None,  chart_instructions: Union[str, None] = None, chart_code:  Union[str, None]=None, plottingLib: str = 'Plotly', **kwargs
    ) -> str:
        if question is not None:
            if chart_instructions is not None:
                question = ( question   + " -- When plotting, follow these instructions: "   + chart_instructions  )
            else:
                question = "When plotting, follow these instructions: " + chart_instructions
            system_msg = f"The following is a pandas DataFrame that contains the results of the query that answers the question the user asked: '{question}'"
        else:
            system_msg = "The following is a pandas DataFrame "

        if sql is not None:
            system_msg += f"\n\nThe DataFrame was produced using this query: {sql}\n\n"

        system_msg += f"The following is information about the resulting pandas DataFrame 'df': \n{df_metadata}"
        
        if chart_code is not None:
            system_msg += f"The following is the Python {plottingLib} code that has already been generated for the resulting pandas DataFrame 'df': \n{chart_code}"

        message_log = [
            self.system_message(system_msg),
            self.user_message(
                f"Can you update the Python {plottingLib} code to meet the user's plotting instructions and chart the results of the dataframe? Assume the data is in a pandas dataframe called 'df'. If there is only one value in the dataframe, use an Indicator. Respond with only Python {plottingLib} code. Do not answer with any explanations -- just the code."
            ),
        ]

        plotly_code = self.submit_prompt(message_log, kwargs=kwargs)

        return self._sanitize_plotly_code(self._extract_python_code(plotly_code))
    
    def connect_to_snowflake(
        self,
        account: str,
        username: str,
        password: str,
        database: str,
        schema: str,
        role: Union[str, None] = None,
        warehouse: Union[str, None] = None,
    ):
        try:
            snowflake = __import__("snowflake.connector")
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method, run command:"
                " \npip install vanna[snowflake]"
            )

        if username == "my-username":
            username_env = os.getenv("SNOWFLAKE_USERNAME")

            if username_env is not None:
                username = username_env
            else:
                raise ImproperlyConfigured("Please set your Snowflake username.")

        if password == "my-password":
            password_env = os.getenv("SNOWFLAKE_PASSWORD")

            if password_env is not None:
                password = password_env
            else:
                raise ImproperlyConfigured("Please set your Snowflake password.")

        if account == "my-account":
            account_env = os.getenv("SNOWFLAKE_ACCOUNT")

            if account_env is not None:
                account = account_env
            else:
                raise ImproperlyConfigured("Please set your Snowflake account.")

        if database == "my-database":
            database_env = os.getenv("SNOWFLAKE_DATABASE")

            if database_env is not None:
                database = database_env
            else:
                raise ImproperlyConfigured("Please set your Snowflake database.")

        conn = snowflake.connector.connect(
            user=username,
            password=password,
            account=account,
            database=database,
            schema=schema,
            role=role,
            warehouse=warehouse
        )

        def run_sql_snowflake(sql: str) -> pd.DataFrame:
            cs = conn.cursor()

            #not needed is setting role, warehouse, database and schema on connection
            # if role is not None:
            #     cs.execute(f"USE ROLE {role}")

            # if warehouse is not None:
            #     cs.execute(f"USE WAREHOUSE {warehouse}")
            # cs.execute(f"USE DATABASE {database}")

            cur = cs.execute(sql)

            results = cur.fetchall()

            # Create a pandas dataframe from the results
            df = pd.DataFrame(results, columns=[desc[0] for desc in cur.description])

            return df

        self.run_sql = run_sql_snowflake
        self.run_sql_is_set = True

    def get_related_documentation(self, question: str, tag=None, **kwargs) -> list:
        query_params = {
            "query_texts": [question],
        }

        # Add the tag to the query if it's provided
        if tag is not None:
            query_params["tag"] = tag

        return ChromaDB_VectorStore._extract_documents(
            self.documentation_collection.query(**query_params)
        )
    
    def runTrainingPlanSnowflake(self):
        plan = self.get_training_plan_snowflake()
        self.train(plan=plan)

    def get_sql_prompt(
        self,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        questionConversationHistory: list,
        questionMemoryLen: int = 2,
        **kwargs,
    ):
        initial_prompt = "The user provides a question and you provide SQL. You will only respond with SQL code and not with any explanations.\n\nRespond with ONLY with SQL code from 'SELECT' to the ';'. In the SQL code, do your best to provide nicely named columns along additional metadata columns to answer the question. Do not answer with any explanations -- just the select statement code 'SELECT' to the ';'.\n"

        initial_prompt = OpenAI_Chat.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=14000
        )

        initial_prompt = OpenAI_Chat.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=14000
        )

        message_log = [OpenAI_Chat.system_message(initial_prompt)]

        for example in question_sql_list:
            if example is None:
                print("example is None")
            else:
                if example is not None and "question" in example and "sql" in example:
                    message_log.append(OpenAI_Chat.user_message(example["question"]))
                    message_log.append(OpenAI_Chat.assistant_message(example["sql"]))

        message_log.append({"role": "user", "content": question})
        msgHistory = questionMemoryLen*-1

        if(questionMemoryLen < len(questionConversationHistory)):
            for message in questionConversationHistory[msgHistory:]:
                if message["type"] =='markdown':
                    message_log.append({"role": message['role'], "content":message["content"]})
                elif message["type"] =='code':
                    message_log.append({"role": message['role'], "content":message["content"]})
                elif message["type"] =='sql':
                    message_log.append({"role": message['role'], "content":message["content"]})
                elif message["type"] =='dataframe':
                    message_log.append({"role": message['role'], "content":message["df"].to_string()})
                elif message["type"] =='sql-dataframe':
                    message_log.append({"role": message['role'], "content":'Here is a the generated SQL query for your question:'})
                    message_log.append({"role": message['role'], "content":message["sql"]})
                    message_log.append({"role": message['role'], "content":'Data Preview (first 5 rows):'})
                    message_log.append({"role": message['role'], "content":message["df"].to_string()})
                elif message["type"] =='figure-code':
                    message_log.append({"role": message['role'], "content":'Here is a figure for the data: <img>'})
                    message_log.append({"role": message['role'], "content":'Here is the code for the figure:'})
                    message_log.append({"role": message['role'], "content":message["code"]})
                elif message["type"] =='figure':
                    message_log.append({"role": message['role'], "content":'<img>'})
                elif  message["type"] =='error':
                    message_log.append({"role": message['role'], "content":message["content"]})
                else:
                    message_log.append({"role": message['role'], "content":message["content"]})

        # print(message_log)
        return message_log
    
    def extract_sql(self, llm_response: str) -> str:
        # If the llm_response contains a markdown code block, with or without the sql tag, extract the sql from it
        sql = re.search(r"```sql\n(.*)```", llm_response, re.DOTALL)
        if sql:
            self.log(f"Output from LLM: {llm_response} \nExtracted SQL: {sql.group(1)}")
            return sql.group(1)
        # Regular expression pattern to match a SQL query
        sql = re.search(r"(SELECT\s+.*?;)", llm_response,  re.IGNORECASE | re.DOTALL)
        if sql:
            self.log(f"Output from LLM: {llm_response} \nExtracted SQL: {sql.group(1)}")
            return sql.group(1)
        
        sql = re.search(r"```(.*)```", llm_response, re.DOTALL)
        if sql:
            self.log(f"Output from LLM: {llm_response} \nExtracted SQL: {sql.group(1)}")
            return sql.group(1)
        
        return llm_response
    
    def generate_sql(self, question: str,questionConversationHistory:list, **kwargs) -> str:
        question_sql_list = self.get_similar_question_sql(question, **kwargs)
        ddl_list = self.get_related_ddl(question, **kwargs)
        doc_list = self.get_related_documentation(question, **kwargs)
        prompt = self.get_sql_prompt(
            question=question,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            questionConversationHistory=questionConversationHistory,
            **kwargs,
        )
        llm_response = self.submit_prompt(prompt, **kwargs)
        # print(llm_response)
        return self.extract_sql(llm_response)
    # def get_similar_question_sql(self, question: str, tag=None, **kwargs) -> list:
    #     query_params = {
    #         "query_texts": [question],
    #     }

    #     # Add the tag to the query if it's provided
    #     if tag is not None:
    #         query_params["tag"] = tag

    #     return ChromaDB_VectorStore._extract_documents(
    #         self.sql_collection.query(
    #             **query_params
    #         )
    #     )

    # def get_related_ddl(self, question: str, tag=None, **kwargs) -> list:
    #     query_params = {
    #         "query_texts": [question],
    #     }

    #     # Add the tag to the query if it's provided
    #     if tag is not None:
    #         query_params["tag"] = tag

    #     return ChromaDB_VectorStore._extract_documents(
    #         self.ddl_collection.query(
    #             **query_params
    #         )
    #     )
