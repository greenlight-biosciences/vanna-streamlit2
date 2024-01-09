import streamlit as st
import time
import streamlit as st
from code_editor import code_editor
#from utils.vanna_calls import *
import streamlit as st
#import vanna as vn
from vanna.openai.openai_chat import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore


class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
        def __init__(self, config=None):
            ChromaDB_VectorStore.__init__(self, config=config)
            OpenAI_Chat.__init__(self, config=config)


vn= MyVanna(config={'api_key': 'XXXXX', 'model': 'gpt-3.5-turbo'})
vn.connect_to_sqlite('biotech_database.db') 


def setup_session_state():
    st.session_state["my_question"] = None

st.set_page_config(layout="wide")


st.sidebar.title("Output Settings")
st.sidebar.checkbox("Show SQL", value=True, key="show_sql")
st.sidebar.checkbox("Show Table", value=True, key="show_table")
st.sidebar.checkbox("Show Plotly Code", value=True, key="show_plotly_code")
st.sidebar.checkbox("Show Chart", value=True, key="show_chart")
st.sidebar.checkbox("Show Follow-up Questions", value=True, key="show_followup")
st.sidebar.button("Rerun", on_click=setup_session_state, use_container_width=True)

st.title("Data G.E.N.I.E")
st.sidebar.write(st.session_state)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.exchangesCounter= 0
    st.session_state.prompt = None
    st.session_state.firstMessage = True
    st.session_state.radio_sql=""

def set_question(question):
    st.session_state['prompt'] = question
    st.session_state['firstMessage'] = False

def submit_chat(author, message):
    st.chat_message(author).markdown(message)
    st.session_state.messages.append({"role":author, "content": message })

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.chat_input("Ask a Question here ...")

# React to user input
if st.session_state['prompt'] != None:
    
    # Display user message in chat message container
    st.chat_message("user").markdown(st.session_state['prompt'])
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": st.session_state['prompt']})

    #response
    # Display assistant response in chat message container
    sql = vn.generate_sql(question=st.session_state['prompt'])
    if sql:
        if st.session_state.get("show_sql", True):
            with st.chat_message("assistant"):
                st.code(sql, language="sql", line_numbers=True)
                st.session_state.messages.append({"role": "assistant", "content": sql})
            with st.chat_message("user"):
                sqlRadioInput = st.radio(
                    "I would like to ...",
                    options=["Edit :pencil2:", "OK :white_check_mark:"],
                    index=None,
                    captions = ["Edit the SQL", "Use generated SQL"],
                    horizontal = True
                )
    
                if sqlRadioInput == "Edit :pencil2:":
                    submit_chat("user", "I would like to Edit the SQL :pencil2:" )
                    # returnMessage = "I would like to Edit the SQL :pencil2:" 
                    # st.chat_message("user").markdown(returnMessage)
                    # st.session_state.messages.append({"role": "user", "content": returnMessage })
        
                    st.warning(
                        "Please update the generated SQL code. Once you're done hit Shift + Enter to submit"
                    )
                    
                    sql_response = code_editor(sql, lang="sql")
                    fixed_sql_query = sql_response["text"]

                    if fixed_sql_query != "":
                        st.session_state.messages.append({"role": "user", "content": ":pencil: SQL: " + fixed_sql_query})
                        df = vn.run_sql(sql=sql)
                    else:
                        df = None
                elif sqlRadioInput == "OK :white_check_mark:":
                    st.session_state.messages.append({"role": "user", "content": "SQL looks good :white_check_mark:" })

                    df = vn.run_sql(sql=sql)
                else:
                    df = None
        else:
            df = vn.run_sql(sql=sql)

        if df is not None:
            st.session_state["df"] = df  
        if st.session_state.get("df") is not None:
            if st.session_state.get("show_table", True):
                df = st.session_state.get("df")
                with st.chat_message("assistant"):
                    if len(df) > 10:
                        st.text("First 10 rows of data")
                        st.dataframe(df.head(10))
                        st.session_state.messages.append({"role": "assistant", "content": "First 10 rows of data"  })
                        st.session_state.messages.append({"role": "assistant", "content": df.head(10)  })
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": df  })
                        st.dataframe(df)
        
        # with user_message_sql_check:
        chart_instructions_input= "Please make the figure red in color and title it Nice Red figure"
        
        # if chart_instructions_input != '':
        code = vn.generate_plotly_code(question=st.session_state['prompt'], sql=sql, df=df,chart_instructions=chart_instructions_input)
        
        # else:
            #    code = vn.generate_plotly_code(question=my_question, sql=sql, df=df)

        if st.session_state.get("show_plotly_code", False):
            with st.chat_message("assistant"):
                st.code(code, language="python", line_numbers=True )
                st.session_state.messages.append({"role": "assistant", "content": code  })
            with st.chat_message("user"):
                plotyRadioInput = st.radio(
                        "I would like to ...",
                        options=["Edit :pencil2:", "OK :white_check_mark:"],
                        index=None,
                        captions = ["Edit the Plot code", "Use generated Plot code"],
                        horizontal = True
                    )
                if plotyRadioInput == "Edit :pencil2:":
                    st.session_state.messages.append({"role": "user", "content": "I would like to Edit the Plot Code :pencil2:"  })

                    st.warning(
                        "Please fix the generated Python code. Once you're done hit Shift + Enter to submit"
                    )
                    python_code_response = code_editor(code, lang="python")
                    code = python_code_response["text"]
                    st.session_state.messages.append({"role": "user", "content": python_code_response })
                elif plotyRadioInput == "K :white_check_mark:":
                    st.session_state.messages.append({"role": "user", "content": "Plot code looks good! :white_check_mark:"  })

            with st.chat_message("assistant"):
                if code is not None and code != "":
                    if st.session_state.get("show_chart", True):
                            fig = vn.get_plotly_figure(plotly_code=code , df=df)
                            if fig is not None:
                                st.plotly_chart(fig)
                                st.session_state.messages.append({"role": "assistant", "content": fig  })

                            else:
                                st.error("I couldn't generate a chart")
                                


elif st.session_state['firstMessage']:
    with st.chat_message("assistant"):
        questions = vn.generate_questions() 

        for i, question in enumerate(questions):
            time.sleep(0.05)
            button = st.button(
                question,
                on_click=set_question,
                args=(question,),
            )
        st.session_state.messages.append({"role": "assistant", "content": "-".join(questions)})
        st.session_state['firstMessage'] = False

    

