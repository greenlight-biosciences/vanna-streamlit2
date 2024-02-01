import streamlit as st
#import vanna as vn
from vanna.openai.openai_chat import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore


class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
        def __init__(self, config=None):
            ChromaDB_VectorStore.__init__(self, config=config)
            OpenAI_Chat.__init__(self, config=config)


vn= MyVanna(config={'api_key': 'XXXXXX', 'model': 'gpt-3.5-turbo'})
vn.connect_to_sqlite('biotech_database.db') 


def setup_session_state():
    st.session_state["my_question"] = None

#@st.cache_data(show_spinner="Generating sample questions ...")
# def generate_questions_cached(vn: MyVanna):
#     return vn.generate_questions()


#@st.cache_data(show_spinner="Generating SQL query ...")
# def generate_sql_cached(question: str,vn: MyVanna):
#     return vn.generate_sql(question=question)


#@st.cache_data(show_spinner="Running SQL query ...")
# def run_sql_cached(sql: str,vn: MyVanna):
#     return vn.run_sql(sql=sql)


#@st.cache_data(show_spinner="Generating Plotly code ...")
# def generate_plotly_code_cached(question, sql, df,vn: MyVanna):
#     code = vn.generate_plotly_code(question=question, sql=sql, df=df)
#     return code


#@st.cache_data(show_spinner="Running Plotly code ...")
# def generate_plot_cached(code, df,vn: MyVanna):
#     return vn.get_plotly_figure(plotly_code=code, df=df)


#@st.cache_data(show_spinner="Generating followup questions ...")
# def generate_followup_cached(question, df,vn: MyVanna):
#     return vn.generate_followup_questions(question=question, df=df)
