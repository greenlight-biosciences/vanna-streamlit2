
from typing import Union, List, Union
from vanna import get_models, set_model
from vanna.openai.openai_chat import OpenAI_Chat
from chromasdb_vector import ChromaDB_VectorStore
import snowflake.connector
import pandas as pd
from vanna.exceptions import DependencyError, ImproperlyConfigured, ValidationError
import os
import re
import plotly
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from bokeh.plotting import figure
from vanna.__init__ import TrainingPlan, TrainingPlanItem
import logging
import streamlit as st

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)
        self.setup_logger()
        #VannaBase.__init__(self, config=config)

    # def get_models():
    #     return get_models()
    
    # def set_model():
    #     return set_model()
    def setup_logger(self):
        # Configure logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Create a logger
        self.logger = logging.getLogger(__name__)
        
    def log(self, message: str):
        self.logger.debug(message)
    def logDebug(self, message: str):
        self.logger.debug(message)
    def logError(self, message: str):
        self.logger.error(message)
    def logWarning(self, message: str):
        self.logger.warning(message)
    def logInfo(self, message: str):
        self.logger.info(message)


    def _sanitize_plot_code(self, raw_plot_code: str) -> str:
        # Remove the fig.show() statement from the plotly code
        raw_plot_code = raw_plot_code.replace("fig.show()", "")
        raw_plot_code = raw_plot_code.replace("show(fig)", "")
        raw_plot_code = raw_plot_code.replace("show(fig)", "")
        plot_code = raw_plot_code.replace("show(p)", "")

        return plot_code
    
    def generate_plot_code(
        self, question: str = None, sql: str = None, df_metadata: str = None,  chart_instructions: Union[str, None] = None, plottingLib: str = 'Plotly', **kwargs
    ) -> str:
        if question is not None:
            if chart_instructions is not None:
                question = ( question   + " -- When plotting, follow these instructions: "   + chart_instructions  )
            else:
                question = "When plotting, follow these instructions: "
            system_msg = f"The following is a pandas DataFrame that contains the results of the query that answers the question the user asked: '{question}'"
        else:
            system_msg = "The following is a pandas DataFrame "

        if sql is not None:
            system_msg += f"\n\nThe DataFrame was produced using this query: {sql}\n\n"

        system_msg += f"The following is information about the resulting pandas DataFrame 'df': \n{df_metadata}"
        system_msg += "\n -IMPORTANT: When referencing the pandas DataFrame, it is CRUCIAL to ensure that column names are matched EXACTLY when generating plot code. This is NON-NEGOTIABLE for accurate data visualization."
        system_msg += "\n -NEVER include an fig.show() or show(fig) in your returns"
        system_msg += "\n -ALWAYS call the plotting figure or chart variable fig"

        message_log = [
            self.system_message(system_msg),
            self.user_message(
            #f"Can you generate the Python {plottingLib} code to chart the results of the dataframe? Assume the data is in a pandas dataframe called 'df'. If there is only one value in the dataframe, use an Indicator. Respond with only Python {plottingLib} code. Do not answer with any explanations -- just the code."

               f"Can you generate the Python {plottingLib} code to chart the results of the dataframe? Assume the data is in a pandas dataframe called 'df'. Respond with only Python {plottingLib} code. Do not answer with any explanations -- just the code."
            ),
        ]
        self.logInfo(message_log)
        plotly_code = self.submit_prompt(message_log, kwargs=kwargs)
        return self._sanitize_plot_code(self._extract_python_code(plotly_code))
    
    def edit_plot_code(
        self, question: str = None, sql: str = None, df_metadata: str = None,  chart_instructions: Union[str, None] = None, chart_code:  Union[str, None]=None, plottingLib: str = 'Plotly', **kwargs
    ) -> str:
        if question is not None:
            if chart_instructions is not None:
                question = ( question   + " -- When plotting, follow these instructions: "   + chart_instructions  )
            else:
                question = "When plotting, follow these instructions: "
            system_msg = f"The following is a pandas DataFrame that contains the results of the query that answers the question the user asked: '{question}'"
        else:
            system_msg = "The following is a pandas DataFrame "

        if sql is not None:
            system_msg += f"\n\nThe DataFrame was produced using this query: {sql}\n\n"

        system_msg += f"The following is information about the resulting pandas DataFrame 'df': \n{df_metadata}"
        
        if chart_code is not None:
            system_msg += f"The following is the Python {plottingLib} code that has already been generated for the resulting pandas DataFrame 'df': \n{chart_code}"
        system_msg += "\n -NEVER include an fig.show() or show(fig) in your returns"
        system_msg += "\n -ALWAYS call the plotting figure or chart variable fig, not p ONLY fig"

        message_log = [
            self.system_message(system_msg),
            self.user_message(
                f"Can you update the Python {plottingLib} code to meet the user's plotting instructions and chart the results of the dataframe? Assume the data is in a pandas dataframe called 'df'. If there is only one value in the dataframe, use an Indicator. Respond with only Python {plottingLib} code. Do not answer with any explanations -- just the code."
            ),
        ]
        self.logInfo(message_log)
        plotly_code = self.submit_prompt(message_log, kwargs=kwargs)

        return self._sanitize_plot_code(self._extract_python_code(plotly_code))
    
    # def promptReview(
    #     self, question: str = None, additionalInstructions:str =None, **kwargs ) -> str:
    #     if question is not None:
    #         if additionalInstructions is not None:
    #             question = ( question   + " -- When Reviewing the Prompt also take into account these instuctions: "   + additionalInstructions  )
    #         else:
    #             question = "When plotting, follow these instructions: "
    #         system_msg = f"The following is a pandas DataFrame that contains the results of the query that answers the question the user asked: '{question}'"
    #     else:
    #         system_msg = "The following is a pandas DataFrame "

    #     system_msg += f"The following is information about the resulting pandas DataFrame 'df':"
    #     system_msg += f"\n -NEVER include an fig.show() or show(fig) in your returns"
    #     system_msg += f"\n -ALWAYS call the plotting figure or chart variable fig"

    #     message_log = [
    #         self.system_message(system_msg),
    #         self.user_message(
    #         #f"Can you generate the Python {plottingLib} code to chart the results of the dataframe? Assume the data is in a pandas dataframe called 'df'. If there is only one value in the dataframe, use an Indicator. Respond with only Python {plottingLib} code. Do not answer with any explanations -- just the code."

    #            f"Can you generate the Python code to chart the results of the dataframe? Assume the data is in a pandas dataframe called 'df'. Respond with only Python {plottingLib} code. Do not answer with any explanations -- just the code."
    #         ),
    #     ]

    #     plotly_code = self.submit_prompt(message_log, kwargs=kwargs)
    #     return self._sanitize_plot_code(self._extract_python_code(plotly_code))
    
    
    def describeSQLData(
        self, question: str = None, sql:str =None, additionalInstructions:str =None, df_describe:str =None, **kwargs ) -> str:
        if question is not None:
            if additionalInstructions is not None:
                question = ( question   + " -- When describing this data take into account these instructions: "   + additionalInstructions  +"\n")
            else:
                question = "Explain the data that is returned"

        system_msg = " - You are a data reviewer and you provide three short helpful and easy to understand bullet point explanation statements for the python pandas DataFrame description that is given"
        system_msg += " - Take into account the users question when forming your explantation"
        system_msg += f" - Here is the question the user asked, that was used to generated the sql and data: '{question}' \n"
        system_msg +=  " - Take into account the SQL query that was generated to answer the users question when forming your explantation"
        system_msg += f" - SQL query used to generate the data and answer the questions is: '{sql}'\n"
        system_msg += f" - Below is the dataframe description from the dataframe which was generated for the question and sql statements above: {df_describe}"
        system_msg += " - For Categorical Data try to give the user a sense of what categories are available in columns"
        system_msg += " - ONLY use simple words, do not mention anything about dataframes or database query jargon when providing an explanation "
        system_msg += " - ENSURE your responses is under 120 words "

        message_log = [
            self.system_message(system_msg),
            self.user_message(
            #f"Can you generate the Python {plottingLib} code to chart the results of the dataframe? Assume the data is in a pandas dataframe called 'df'. If there is only one value in the dataframe, use an Indicator. Respond with only Python {plottingLib} code. Do not answer with any explanations -- just the code."

               "Provide an explanation for the data as it pertains to my question"
            ),
        ]
        #self.logDebug(message_log)
        sqlDataDesc = self.submit_prompt(message_log, kwargs=kwargs)
        return sqlDataDesc
    
    # @st.cache_resource
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

            if role is not None:
                roleSQL = f"USE ROLE {role}"
                cs.execute(roleSQL)

            if warehouse is not None:
                warehouseSQL= f"USE WAREHOUSE {warehouse}"
                cs.execute(warehouseSQL)
            
            dbuseSQL= f"USE DATABASE {database}"
            cs.execute(dbuseSQL)

            dbuseschema= f"USE SCHEMA {schema}"
            cs.execute(dbuseschema)
            
            cur = cs.execute(sql)

            results = cur.fetchall()

            # Create a pandas dataframe from the results
            df = pd.DataFrame(results, columns=[desc[0] for desc in cur.description])

            return df

        self.run_sql = run_sql_snowflake
        self.run_sql_is_set = True
    
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
        initial_prompt = "The user provides a question and you provide SQL code to run on Snowflake database. You will only respond with SQL code and not with any explanations.\n\nRespond with ONLY with SQL code, you may respond using a 'SELECT' or 'WITH' for complex queries, ending you responses at ';'. In the SQL code, do your best to provide nicely named columns along additional metadata columns to answer the question. Do not answer with any explanations -- just the sql code from 'SELECT' or 'WITH' to the ';'.\n"

        initial_prompt = OpenAI_Chat.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=14000
        )

        initial_prompt = OpenAI_Chat.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=14000
        )

        message_log = [OpenAI_Chat.system_message(initial_prompt)]

        for example in question_sql_list:
            if example is None:
                self.logInfo("example is None")
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
                    message_log.append({"role": message['role'], "content":message["sql"]})
                elif message["type"] =='dataframe':
                    message_log.append({"role": message['role'], "content":message["df"].head(int(message["nrows"])).to_string()})
                elif message["type"] =='sql-dataframe':
                    message_log.append({"role": message['role'], "content":'Here is a the generated SQL query for your question:'})
                    message_log.append({"role": message['role'], "content":message["sql"]})
                    message_log.append({"role": message['role'], "content":'Data Preview (first 5 rows):'})
                    message_log.append({"role": message['role'], "content":message["df"].head(int(message["nrows"])).to_string()})
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

        self.logInfo(message_log)
        return message_log
    
    def extract_sql(self, llm_response: str) -> str:
        # If the llm_response contains a markdown code block, with or without the sql tag, extract the sql from it
        sql = re.search(r"```sql\n(.*)```", llm_response, re.DOTALL)
        if sql:
            self.logDebug(f"Output from LLM: {llm_response} \nExtracted SQL: {sql.group(1)}")
            return sql.group(1)
        # Regular expression pattern to match a SQL query
        sql = re.search(r"((WITH\s+.+?\s+AS\s.+)|(SELECT\s+.+?));", llm_response,  re.IGNORECASE | re.DOTALL)
        if sql:
            self.logDebug(f"Output from LLM: {llm_response} \nExtracted SQL: {sql.group(1)}")
            return sql.group(1)
        
        sql = re.search(r"```(.*)```", llm_response, re.DOTALL)
        if sql:
            self.logDebug(f"Output from LLM: {llm_response} \nExtracted SQL: {sql.group(1)}")
            return sql.group(1)
        
        return llm_response
    
    def generate_sql(self, question: str,questionConversationHistory:list, schema:str=None, **kwargs) -> str:
        question_sql_list = self.get_similar_question_sql(question, schema=schema, **kwargs)
        ddl_list = self.get_related_ddl(question,schema=schema, **kwargs)
        doc_list = self.get_related_documentation(question,schema=schema, **kwargs)
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
    

    def get_plotly_figure(self, plotly_code: str, df: pd.DataFrame, dark_mode: bool = True) -> plotly.graph_objs.Figure:
        """
        **Example:**
        ```python
        fig = vn.get_plotly_figure(
            plotly_code="fig = px.bar(df, x='name', y='salary')",
            df=df
        )
        fig.show()
        ```
        Get a Plotly figure from a dataframe and Plotly code.

        Args:
            df (pd.DataFrame): The dataframe to use.
            plotly_code (str): The Plotly code to use.

        Returns:
            plotly.graph_objs.Figure: The Plotly figure.
        """
        ldict = {"df": df, "px": px, "go": go}
        fig = None
        error = None
        try:
            exec(plotly_code, globals(), ldict)

            fig = ldict.get("fig", None)
            if fig is None:
                raise Exception('Failed to generate Figure')
        except Exception as e:
            self.logError(e)
            error =e
            # Inspect data types
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            fig = go.Figure()
            fig.add_annotation(text="Error in creating the figure", xref="paper", yref="paper", showarrow=False)
            # # Decision-making for plot type
            # if len(numeric_cols) >= 2:
            #     # Use the first two numeric columns for a scatter plot
            #     fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
            # elif len(numeric_cols) == 1 and len(categorical_cols) >= 1:
            #     # Use a bar plot if there's one numeric and one categorical column
            #     fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0])
            # elif len(categorical_cols) >= 1 and df[categorical_cols[0]].nunique() < 10:
            #     # Use a pie chart for categorical data with fewer unique values
            #     fig = px.pie(df, names=categorical_cols[0])
            # else:
            #     # Default to a simple line plot if above conditions are not met
            #     fig = px.line(df)

        if fig is None:
            return None

        if dark_mode:
            fig.update_layout(template="plotly_dark")

        return fig, error
    
    def get_altair_figure(self, altair_code: str, df: pd.DataFrame, dark_mode: bool = True):
        """
        Get an Altair chart from a dataframe and Altair code.

        Args:
            df (pd.DataFrame): The dataframe to use.
            altair_code (str): The Altair code to use.

        Returns:
            alt.Chart: The Altair chart.
        """
        ldict = {"df": df, "alt": alt}
        fig = None
        error = None
        try:
            exec(altair_code, globals(), ldict)
            fig = ldict.get("fig", None)
            if fig is None:
                fig = ldict.get("chart", None)
                if fig is None:
                    raise Exception('Failed to generate Figure')
        except Exception as e:
            self.logError(e)
            error = e
            fig = alt.Chart(df).mark_text().encode(
                text=alt.Text(value='Error in creating the chart')
            )

        return fig, error
    
    def get_bokeh_figure(self, bokeh_code: str, df: pd.DataFrame, dark_mode: bool = True):
        """
        Get a Bokeh figure from a dataframe and Bokeh code.

        Args:
            df (pd.DataFrame): The dataframe to use.
            bokeh_code (str): The Bokeh code to use.

        Returns:
            bokeh.plotting.figure.Figure: The Bokeh figure.
        """
        ldict = {"df": df, "figure": figure}
        fig = None
        error = None
        try:
            exec(bokeh_code, globals(), ldict)
            fig = ldict.get("fig", None)
            if fig is None:
                fig = ldict.get("p", None)
                if fig is None:
                    raise Exception('Failed to generate Figure')
        except Exception as e:
            self.logError(e)
            error =e
            fig = figure(title="Error in creating the figure")
            fig.text(x=0, y=0, text=['Error'])

        return fig, error

    def train(
        self,
        question: str = None,
        sql: str = None,
        ddl: str = None,
        documentation: str = None,
        plan: TrainingPlan = None,
        schema:str = None,
    ) -> str:
        """
        **Example:**
        ```python
        vn.train()
        ```

        Train Vanna.AI on a question and its corresponding SQL query.
        If you call it with no arguments, it will check if you connected to a database and it will attempt to train on the metadata of that database.
        If you call it with the sql argument, it's equivalent to [`add_sql()`][vanna.add_sql].
        If you call it with the ddl argument, it's equivalent to [`add_ddl()`][vanna.add_ddl].
        If you call it with the documentation argument, it's equivalent to [`add_documentation()`][vanna.add_documentation].
        Additionally, you can pass a [`TrainingPlan`][vanna.TrainingPlan] object. Get a training plan with [`vn.get_training_plan_experimental()`][vanna.get_training_plan_experimental].

        Args:
            question (str): The question to train on.
            sql (str): The SQL query to train on.
            ddl (str):  The DDL statement.
            documentation (str): The documentation to train on.
            plan (TrainingPlan): The training plan to train on.
        """

        if question and not sql:
            raise ValidationError("Please also provide a SQL query")

        if documentation:
            self.logInfo("Adding documentation....")
            return self.add_documentation(documentation,schema=schema)

        if sql:
            if question is None:
                question = self.generate_question(sql)
                self.logInfo("Question generated with sql:", question, "\nAdding SQL...")
            return self.add_question_sql(question=question, sql=sql,schema=schema)

        if ddl:
            self.logInfo("Adding ddl:", ddl)
            return self.add_ddl(ddl,schema=schema)

        if plan:
            for item in plan._plan:
                if item.item_type == TrainingPlanItem.ITEM_TYPE_DDL:
                    self.add_ddl(item.item_value,schema=schema)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_IS:
                    self.add_documentation(item.item_value,schema=schema)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_SQL:
                    self.add_question_sql(question=item.item_name, sql=item.item_value, schema=schema)
    
    def _get_databases(self) -> List[str]:
        try:
            self.logInfo("Trying INFORMATION_SCHEMA.DATABASES")
            df_databases = self.run_sql("SELECT * FROM INFORMATION_SCHEMA.DATABASES")
        except Exception as e:
            self.logError(e)
            try:
                self.logError("Trying SHOW DATABASES")
                df_databases = self.run_sql("SHOW DATABASES")
            except Exception as e:
                self.logError(e)
                return []

        return df_databases["DATABASE_NAME"].unique().tolist()                
    def trainVN(self, input , type, question =None,schema:str=None):
        if type =='ddl':
            return self.train(ddl=input,schema=schema)
        elif type =='doc':
            return self.train(documentation=input,schema=schema)
        elif type =='sql':
            # Check if question is provided
            if question:
               return  self.train(sql=input, question=question,schema=schema)
            else:
                return self.train(sql=input,schema=schema) 
    
    def _get_information_schema_tables(self, database: str, schema) -> pd.DataFrame:
        df_tables = self.run_sql(f"SELECT * FROM {database}.INFORMATION_SCHEMA.TABLES where TABLE_SCHEMA ='{schema}'")
        return df_tables
    
    def get_training_plan_snowflake(
        self,
        filter_databases: Union[List[str], None] = None,
        filter_schemas: Union[List[str], None] = None,
        include_information_schema: bool = False,
        use_historical_queries: bool = True,
    ) -> TrainingPlan:
        plan = TrainingPlan([])

        if self.run_sql_is_set is False:
            raise ImproperlyConfigured("Please connect to a database first.")

        if use_historical_queries:
            try:
                self.logInfo("Trying query history")
                df_history = self.run_sql(
                    """ select * from table(information_schema.query_history(result_limit => 5000)) order by start_time"""
                )

                df_history_filtered = df_history.query("ROWS_PRODUCED > 1")
                if filter_databases is not None:
                    mask = (
                        df_history_filtered["QUERY_TEXT"]
                        .str.lower()
                        .apply(
                            lambda x: any(
                                s in x for s in [s.lower() for s in filter_databases]
                            )
                        )
                    )
                    df_history_filtered = df_history_filtered[mask]

                if filter_schemas is not None:
                    mask = (
                        df_history_filtered["QUERY_TEXT"]
                        .str.lower()
                        .apply(
                            lambda x: any(
                                s in x for s in [s.lower() for s in filter_schemas]
                            )
                        )
                    )
                    df_history_filtered = df_history_filtered[mask]

                if len(df_history_filtered) > 10:
                    df_history_filtered = df_history_filtered.sample(10)

                for query in df_history_filtered["QUERY_TEXT"].unique().tolist():
                    plan._plan.append(
                        TrainingPlanItem(
                            item_type=TrainingPlanItem.ITEM_TYPE_SQL,
                            item_group="",
                            item_name=self.generate_question(query),
                            item_value=query,
                        )
                    )

            except Exception as e:
                self.logError(e)

        databases = self._get_databases()

        for database in databases:
            if filter_databases is not None and database not in filter_databases:
                continue

            try:
                df_tables = self._get_information_schema_tables(database=database, schema=filter_schemas)

                self.logInfo(f"Trying INFORMATION_SCHEMA.COLUMNS for {database}")
                df_columns = self.run_sql(
                    f"SELECT * FROM {database}.INFORMATION_SCHEMA.COLUMNS where TABLE_SCHEMA ='{filter_schemas}'"
                )

                self.logInfo(f'Filtered Schemas {filter_schemas}')
      
                for schema in df_tables["TABLE_SCHEMA"].unique().tolist():
                    self.logInfo(f'Trying Schema {filter_schemas}')
                    if filter_schemas is not None and schema not in filter_schemas:
                        self.logInfo(f'Filtered Schemas {filter_schemas}')
                        continue

                    if (
                        not include_information_schema
                        and schema == "INFORMATION_SCHEMA"
                    ):
                        continue

                    df_columns_filtered_to_schema = df_columns.query(
                        f"TABLE_SCHEMA == '{schema}'"
                    )

                    try:
                        tables = (
                            df_columns_filtered_to_schema["TABLE_NAME"]
                            .unique()
                            .tolist()
                        )

                        for table in tables:
                            df_columns_filtered_to_table = (
                                df_columns_filtered_to_schema.query(
                                    f"TABLE_NAME == '{table}'"
                                )
                            )
                            doc = f"The following columns are in the {table} table in the {database} database:\n\n"
                            doc += df_columns_filtered_to_table[
                                [
                                    "TABLE_CATALOG",
                                    "TABLE_SCHEMA",
                                    "TABLE_NAME",
                                    "COLUMN_NAME",
                                    "DATA_TYPE",
                                    "COMMENT",
                                ]
                            ].to_markdown()

                            plan._plan.append(
                                TrainingPlanItem(
                                    item_type=TrainingPlanItem.ITEM_TYPE_IS,
                                    item_group=f"{database}.{schema}",
                                    item_name=table,
                                    item_value=doc,
                                )
                            )

                    except Exception as e:
                        self.logError(e)
                        pass
            except Exception as e:
                self.logError(e)

        return plan
            
    def runTrainingPlanSnowflake(self, schema:str=None, database:str=None):
        plan = self.get_training_plan_snowflake(filter_databases= database,
                                                 filter_schemas=schema, 
                                                 include_information_schema=True,
                                                 use_historical_queries = False)
        self.logInfo(f"Running Training Plan for DB:{database}")
        self.logInfo(f"Running Training Plan for Schema {schema}")

        self.logInfo(plan)
        self.train(plan=plan,schema=schema)