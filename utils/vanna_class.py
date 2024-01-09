import streamlit as st
#import vanna as vn
from vanna.openai.openai_chat import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore

class MyStreamlitApp:
    def __init__(self):
        self.vn = None
        self.setup_connexion()
        self.setup_session_state()

    @st.cache_resource(ttl=3600)
    def setup_connexion(self):
        class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
            def __init__(self, config=None):
                ChromaDB_VectorStore.__init__(self, config=config)
                OpenAI_Chat.__init__(self, config=config)

        self.vn = MyVanna(config={'api_key': 'your-api-key', 'model': 'gpt-3.5-turbo'})
        self.vn.connect_to_sqlite('biotech_database.db')

    def setup_session_state(self):
        st.session_state["my_question"] = None

    @st.cache_data(show_spinner="Generating sample questions ...")
    def generate_questions_cached(self):
        return self.vn.generate_questions()

    @st.cache_data(show_spinner="Generating SQL query ...")
    def generate_sql_cached(self, question: str):
        return self.vn.generate_sql(question=question)

    @st.cache_data(show_spinner="Running SQL query ...")
    def run_sql_cached(self, sql: str):
        return self.vn.run_sql(sql=sql)

    @st.cache_data(show_spinner="Generating Plotly code ...")
    def generate_plotly_code_cached(self, question, sql, df):
        code = self.vn.generate_plotly_code(question=question, sql=sql, df=df)
        return code

    @st.cache_data(show_spinner="Running Plotly code ...")
    def generate_plot_cached(self, code, df):
        return self.vn.get_plotly_figure(plotly_code=code, df=df)

    @st.cache_data(show_spinner="Generating followup questions ...")
    def generate_followup_cached(self, question, df):
        return self.vn.generate_followup_questions(question=question, df=df)
