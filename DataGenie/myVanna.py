
from typing import Union, List
# from vanna import get_models, set_model
from vanna.openai.openai_chat import OpenAI_Chat
# from chromasdb_vector import ChromaDB_VectorStore
# import snowflake.connector
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
# import streamlit as st
from utility import returnMsgFrmtForOAI
from cosmosVectorStore import AzureCosmosMongovCoreDB
from VannaLogger import VannaLogger
from SnowflakeConnection import SnowflakeConnection
import openai
class MyVanna(AzureCosmosMongovCoreDB, OpenAI_Chat):
    def __init__(self, config=None):
        # ChromaDB_VectorStore.__init__(self, config=config)
        AzureCosmosMongovCoreDB.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)
        self.vannaLogger = VannaLogger(env_var_name='LOG_LEVEL')
        self.snow = SnowflakeConnection()
        self.run_sql_is_set = True
        
    def run_sql(self,sql:str) -> pd.DataFrame:
        with self.snow as conn:
            cur = conn.cursor()
            try:
                cur.execute(sql)
                results= cur.fetchall()
                df = pd.DataFrame(results, columns=[desc[0] for desc in cur.description])
                return df
            finally:
                cur.close()


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
        system_msg += "\n -IMPORTANT: When referencing the pandas DataFrame 'df', it is CRUCIAL to ensure that column names are matched EXACTLY when generating plot code. This is NON-NEGOTIABLE for accurate data visualization."
        system_msg += "\n -NEVER include an fig.show() or show(fig) in your returns"
        system_msg += "\n -ALWAYS call the plotting figure or chart variable fig"

        message_log = [
            self.system_message(system_msg),
            self.user_message(
            #f"Can you generate the Python {plottingLib} code to chart the results of the dataframe? Assume the data is in a pandas dataframe called 'df'. If there is only one value in the dataframe, use an Indicator. Respond with only Python {plottingLib} code. Do not answer with any explanations -- just the code."

               f"Can you generate the Python {plottingLib} code to chart the results of the DataFrame 'df'? Assume the data is in a pandas dataframe called 'df' and each column name is UPPERCASED. Respond with only Python {plottingLib} code. Do not answer with any explanations -- just the code."
            ),
        ]
        self.vannaLogger.logInfo(message_log)
        plotly_code = self.submit_prompt(message_log, kwargs=kwargs, model_over_ride='gpt-35-turbo-16k')
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
        self.vannaLogger.logInfo(message_log)
        plotly_code = self.submit_prompt(message_log, kwargs=kwargs, model_over_ride='gpt-35-turbo-16k')

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
        #self.vannaLogger.logDebug(message_log)
        sqlDataDesc = self.submit_prompt(message_log, kwargs=kwargs,model_over_ride='gpt-35-turbo-16k')
        return sqlDataDesc
    
    def get_sql_prompt(
        self,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        questionConversationHistory: list,
        questionMemoryLen: int = 5,
        **kwargs,
    ):
        initial_prompt = "Respond to queries with SQL code for Snowflake database execution only. No introductions, summaries, or any text outside SQL."
        initial_prompt += " Directly begin with 'SELECT' or 'WITH' for complex queries, ending strictly with a semicolon (';').\n"
        initial_prompt += "Column names must be descriptive, including necessary metadata for answering the question directly."
        initial_prompt += " Absolutely no text outside the SQL query, including no preambles like 'Here is the generated SQL query for your question:', is allowed.\n"

        initial_prompt = OpenAI_Chat.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=14000
        )

        initial_prompt = OpenAI_Chat.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=14000
        )

        message_log = [OpenAI_Chat.system_message(initial_prompt)]

        for example in question_sql_list:
            if example is None:
                self.vannaLogger.logInfo("example is None")
            else:
                if example is not None and "question" in example and "sql" in example:
                    message_log.append(OpenAI_Chat.user_message(example["question"]))
                    message_log.append(OpenAI_Chat.assistant_message(example["sql"]))
                else:
                    Exception(f"SQL Question and Answer not in expected format: Q&A SQL: {example}")

        msgHistoryLimit = questionMemoryLen

        if questionConversationHistory:
            startIndex = -msgHistoryLimit if len(questionConversationHistory) > msgHistoryLimit else -len(questionConversationHistory)
            for message in questionConversationHistory[startIndex:]:
                message_log = returnMsgFrmtForOAI(message=message,message_log=message_log)
        self.vannaLogger.logInfo(message_log)
        return message_log
    
    def extract_sql(self, llm_response: str) -> str:
        # If the llm_response contains a markdown code block, with or without the sql tag, extract the sql from it
        sql = re.search(r"```sql\n(.*)```", llm_response, re.DOTALL)
        if sql:
            self.vannaLogger.logDebug(f"Output from LLM: {llm_response} \nExtracted SQL: {sql.group(1)}")
            return sql.group(1)
        # Regular expression pattern to match a SQL query
        sql = re.search(r"((WITH\s+.+?\s+AS\s.+)|(SELECT\s+.+?));", llm_response,  re.IGNORECASE | re.DOTALL)
        if sql:
            self.vannaLogger.logDebug(f"Output from LLM: {llm_response} \nExtracted SQL: {sql.group(1)}")
            return sql.group(1)
        
        sql = re.search(r"```(.*)```", llm_response, re.DOTALL)
        if sql:
            self.vannaLogger.logDebug(f"Output from LLM: {llm_response} \nExtracted SQL: {sql.group(1)}")
            return sql.group(1)
        
        return llm_response
    
    def generate_sql(self, question: str,questionConversationHistory:list, schema:str=None, **kwargs) -> str:
        questionVector=self.generate_embedding(question)
        question_sql_list = self.get_similar_question_sql(question=None,questionVectorIN=questionVector, schema=schema, **kwargs)
        ddl_list = self.get_related_ddl(question=None, questionVectorIN=questionVector,schema=schema, **kwargs)
        doc_list = self.get_related_documentation(question=None,questionVectorIN=questionVector, schema=schema, **kwargs)
        prompt = self.get_sql_prompt(
            question=question,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            questionConversationHistory=questionConversationHistory,
            questionMemoryLen = int(os.environ.get("QUESTIONMEMORYLEN",10)),
            **kwargs,
        )
        llm_response = self.submit_prompt(prompt, **kwargs)
        self.vannaLogger.logInfo(f'Responses from LLM:{llm_response}')
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
            self.vannaLogger.logError(e)
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
            self.vannaLogger.logError(e)
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
            self.vannaLogger.logError(e)
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
            self.vannaLogger.logInfo("Adding documentation....")
            return self.add_documentation(documentation,schema=schema)

        if sql:
            if question is None:
                question = self.generate_question(sql)
                self.vannaLogger.logInfo("Question generated with sql:", question, "\nAdding SQL...")
            return self.add_question_sql(question=question, sql=sql,schema=schema)

        if ddl:
            self.vannaLogger.logInfo("Adding ddl...")
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
            self.vannaLogger.logInfo("Trying INFORMATION_SCHEMA.DATABASES")
            df_databases = self.run_sql("SELECT * FROM INFORMATION_SCHEMA.DATABASES")
        except Exception as e:
            self.vannaLogger.logError(e)
            try:
                self.vannaLogger.logError("Trying SHOW DATABASES")
                df_databases = self.run_sql("SHOW DATABASES")
            except Exception as e:
                self.vannaLogger.logError(e)
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
                self.vannaLogger.logInfo("Trying query history")
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
                self.vannaLogger.logError(e)

        databases = self._get_databases()

        for database in databases:
            if filter_databases is not None and database not in filter_databases:
                continue

            try:
                df_tables = self._get_information_schema_tables(database=database, schema=filter_schemas)

                self.vannaLogger.logInfo(f"Trying INFORMATION_SCHEMA.COLUMNS for {database}")
                df_columns = self.run_sql(
                    f"SELECT * FROM {database}.INFORMATION_SCHEMA.COLUMNS where TABLE_SCHEMA ='{filter_schemas}'"
                )

                self.vannaLogger.logInfo(f'Filtered Schemas {filter_schemas}')
      
                for schema in df_tables["TABLE_SCHEMA"].unique().tolist():
                    self.vannaLogger.logInfo(f'Trying Schema {filter_schemas}')
                    if filter_schemas is not None and schema not in filter_schemas:
                        self.vannaLogger.logInfo(f'Filtered Schemas {filter_schemas}')
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
                            doc = f"The following columns are in the {database}.{schema}.{table} table, format: database.schema.table:\n\n"
                            doc += df_columns_filtered_to_table[
                                [
                                    "TABLE_CATALOG",
                                    "TABLE_SCHEMA",
                                    "TABLE_NAME",
                                    "COLUMN_NAME",
                                    "DATA_TYPE",
                                    # "COMMENT",
                                ]
                            ].to_markdown(index=False)

                            plan._plan.append(
                                TrainingPlanItem(
                                    item_type=TrainingPlanItem.ITEM_TYPE_IS,
                                    item_group=f"{database}.{schema}",
                                    item_name=table ,
                                    item_value=doc,
                                )
                            )

                    except Exception as e:
                        self.vannaLogger.logError(e)
                        pass
            except Exception as e:
                self.vannaLogger.logError(e)

        return plan
            
    def runTrainingPlanSnowflake(self, schema:str=None, database:str=None):
        plan = self.get_training_plan_snowflake(filter_databases= database,
                                                 filter_schemas=schema, 
                                                 include_information_schema=True,
                                                 use_historical_queries = False)
        self.vannaLogger.logInfo(f"Running Training Plan for DB:{database}")
        self.vannaLogger.logInfo(f"Running Training Plan for Schema {schema}")

        self.vannaLogger.logInfo(plan)
        self.train(plan=plan,schema=schema)
    
    def summarizePrompt(
        self, question: str = None, questionConversationHistory:list = None, additionalInstructions:str=None,questionMemoryLen:int=10, **kwargs ) -> str:
        # if question is not None:
        #     # if additionalInstructions is not None:
        #     #     question = ( "Master Question: " + question   + ", also follow these additional intructions "   + additionalInstructions  +"\n")
        #     # else:
        #     #     question = ( "Master Question: " + question )

        try:
            system_msg ="- Your task is to precisely REPHRASE the user's Master question to focus on DATA EXTRACTION from the DATA WAREHOUSE, integrating the latest instructions from the conversation."
            system_msg +="Ensure the rephrased question maintains the essence of the Master query but is explicitly directed towards obtaining specific data or insights from the data warehouse."
            system_msg +="Incorporate all relevant updates, such as schema filters, query adjustments, or particular data extraction requests, ensuring these modifications align with the goal of data retrieval from the data warehouse."
            system_msg +="AVOID adding extraneous explanations or diverging from the data extraction focus. Your refinement should directly support the user's intent to extract data, using clear, concise, and relevant language."
            system_msg +="The updated question should not only reflect the conversation's evolution but also emphasize the user's objective of data extraction, ensuring clarity in the request for specific information or analysis from the data warehouse."
            system_msg += f"-> THIS is the Master Question which needs to be updated: {question}"
            
            message_log = [
                self.system_message(system_msg)           
            ]

            msgHistoryLimit = questionMemoryLen

            if questionConversationHistory:
                startIndex = -msgHistoryLimit if len(questionConversationHistory) > msgHistoryLimit else -len(questionConversationHistory)
                self.vannaLogger.logInfo(questionConversationHistory[startIndex:])

                for message in questionConversationHistory[startIndex:]:
                    self.vannaLogger.logInfo(message)
                    message_log = returnMsgFrmtForOAI(message=message,message_log=message_log)
            
            message_log.append(self.user_message(
                f"Please re-summarize my Master Question based on the preceeding conversation."))
            
            self.vannaLogger.logInfo(message_log)
            #self.vannaLogger.logDebug(message_log)
            summerizedQuestion = self.submit_prompt(message_log, kwargs=kwargs, model_over_ride='gpt-35-turbo-16k')
            return summerizedQuestion
        except Exception as e:
            self.vannaLogger.logError(str(e))
            return question 
         
    def submit_prompt(self, prompt, model_over_ride:str=None, **kwargs) -> str:
        if prompt is None:
            raise Exception("Prompt is None")

        if len(prompt) == 0:
            raise Exception("Prompt is empty")

        # Assuming prompt is an iterable of messages (e.g., list of dicts)
        num_tokens = sum(len(message["content"]) / 4 for message in prompt)  # Simplified token count

        if model_over_ride is not None:
            self.vannaLogger.logInfo(f"Using model {self.config['model']} for {num_tokens} tokens (approx)")
            response = openai.chat.completions.create(
                model=model_over_ride,
                messages=prompt,
                max_tokens=1024,
                stop=None,
                temperature=0.7,
                stream=False
            )
        elif self.config and "engine" in self.config:
            self.vannaLogger.logInfo(f"Using engine {self.config['engine']} for {num_tokens} tokens (approx)")
            response = openai.chat.completions.create(
                engine=self.config["engine"],
                messages=prompt,
                max_tokens=500,
                stop=None,
                temperature=0.7,
                stream=False
            )
        elif self.config and "model" in self.config:
            self.vannaLogger.logInfo(f"Using model {self.config['model']} for {num_tokens} tokens (approx)")
            response = openai.chat.completions.create(
                model=self.config["model"],
                messages=prompt,
                max_tokens=1024,
                stop=None,
                temperature=0.7,
                stream=False
            )
        else:
            # Default model selection logic
            model = "gpt-3.5-turbo-16k" if num_tokens > 3500 else "gpt-3.5-turbo"
            self.vannaLogger.logInfo(f"Using model {model} for {num_tokens} tokens (approx)")
            response = openai.chat.completions.create(
                model=model, messages=prompt, max_tokens=500, stop=None, temperature=0.7, stream=False
            )
        self.vannaLogger.logDebug(f'FULL RESPONSE FROM LLM:{response}')
        # Combine all text responses
        combined_text = response.choices[0].message.content

        if not combined_text:
            # Fallback in case of no text responses
            return "No response from Data G.E.N.I.E - please try again.."

        return combined_text