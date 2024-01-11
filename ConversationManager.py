import streamlit as st
import time
import streamlit as st
from code_editor import code_editor
#from utils.vanna_calls import *
import streamlit as st
#import vanna as vn
from vanna.openai.openai_chat import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from dotenv import load_dotenv
import os
load_dotenv(".env")

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
        def __init__(self, config=None):
            ChromaDB_VectorStore.__init__(self, config=config)
            OpenAI_Chat.__init__(self, config=config)

vn= MyVanna(
    config={	
        'api_type': 'azure',
        'api_base': os.environ.get("AZURE_OPENAI_ENDPOINT"),
	    'api_version': '2023-05-15',
	    #'engine': "gpt-4",
        'model': "gpt-35-turbo",
	    'api_key': os.environ.get("AZUREOPENAIKEY"),
})
vn.connect_to_sqlite('biotech_database.db') 


def setup_session_state():
    st.session_state["my_question"] = None
my_question = st.session_state.get("my_question", default=None)

st.set_page_config(layout="wide")


st.sidebar.title("Output Settings")
st.sidebar.checkbox("Show SQL", value=True, key="show_sql")
st.sidebar.checkbox("Show Table", value=True, key="show_table")
st.sidebar.checkbox("Show Plotly Code", value=True, key="show_plotly_code")
st.sidebar.checkbox("Show Chart", value=True, key="show_chart")
st.sidebar.checkbox("Show Follow-up Questions", value=True, key="show_followup")
st.sidebar.button("Rerun", on_click=setup_session_state, use_container_width=True)

st.title("Data G.E.N.I.E")
def resetPrompt():
    st.session_state['prompt'] = None
    st.session_state['sql'] =None
    st.session_state['code'] =None
    st.session_state['df'] =None
    st.session_state['fig'] =None
    st.session_state['tempSQL'] =None
    st.rerun()
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help?" , "type":"markdown"}]
    st.session_state.prompt = None
    st.session_state.sql =None
    st.session_state.code =None
    st.session_state.df =None
    st.session_state.fig =None
    st.session_state.tempSQL =None
    st.session_state.firstMessage =None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["type"] =='markdown':
            st.markdown(message["content"])
        elif message["type"] =='code':
            st.code(message["content"], language="python", line_numbers=True )
        elif message["type"] =='sql':
            st.code(message["content"], language="sql", line_numbers=True)
        elif message["type"] =='dataframe':
            st.dataframe(message["content"])
        elif message["type"] =='figure':
            st.plotly_chart(message["content"])
        elif  message["type"] =='error':
            st.error(message["content"])
        else:
            st.text(message["content"])
st.sidebar.write(st.session_state)

# def set_question(question):
#     st.session_state['prompt'] = question
#     st.session_state['firstMessage'] = False

# def submit_chat(author, message):
#     st.chat_message(author).markdown(message)
#     st.session_state.messages.append({"role":author, "content": message })

# if st.session_state['prompt'] is None:
#     myQuestion =  st.chat_input( "Ask me a question about your data", )
#     st.session_state['firstMessage']=True
# else:
#     st.session_state['firstMessage']=False
# st.chat_input( "Ask me a question about your data", )
if   myQuestion :=  st.chat_input( "Ask me a question about your data", ) :
    print('entering 1')
    # with st.chat_message("assistant"):
    #     questions = vn.generate_questions() 

    #     for i, question in enumerate(questions):
    #         #time.sleep(0.05)
    #         button = st.button(
    #             question,
    #             on_click=set_question,
    #             args=(question,),
    #         )
    #     st.session_state.messages.append({"role": "assistant", "content": "\n-".join(questions),'type': 'markdown'})
    st.session_state.messages.append({"role": "user", "content": myQuestion, "type":"markdown"})
    st.session_state['prompt'] =myQuestion
    st.session_state['firstMessage']=False
    st.rerun()

elif st.session_state['prompt'] is not None and st.session_state['tempSQL'] is None:   
    print('entering 2') 

    st.session_state['tempSQL']= vn.generate_sql(question=st.session_state['prompt'])

    if st.session_state.get("show_sql", True):
        st.session_state.messages.append({"role": "assistant", "content": st.session_state['tempSQL'] , "type":"sql"})
    else:
        st.session_state['sql'] = st.session_state['tempSQL']
        st.session_state["df"] = vn.run_sql(sql=st.session_state['sql'])
    st.rerun()

elif st.session_state['tempSQL'] is not None and st.session_state['sql'] is None:
    print('entering 3') 
    with st.chat_message("user"):
        sqlRadioInput = st.radio(
            "I would like to ...",
            options=["Edit :pencil2:", "OK :white_check_mark:"],
            index=None,
            captions = ["Edit the SQL", "Use generated SQL"],
            horizontal = True
        )

    if sqlRadioInput == "Edit :pencil2:":
        st.session_state.messages.append({"role": "user", "content": "I would like to Edit the SQL :pencil2:" , "type":"markdown"  })

        st.warning("Please update the generated SQL code. Once you're done hit Shift + Enter to submit" )
        
        sql_response = code_editor(st.session_state['tempSQL'], lang="sql")
        fixed_sql_query = sql_response["text"]

        if fixed_sql_query != "":
            st.session_state.messages.append({"role": "user", "content": ":pencil: SQL: ", "type":"markdown"})
            st.session_state.messages.append({"role": "user", "content":  fixed_sql_query , "type":"sql"})
            st.session_state['sql'] = fixed_sql_query
            st.session_state["df"] = vn.run_sql(sql=st.session_state['sql'])
        else:
            st.session_state['sql'] = None
        st.rerun()
    elif sqlRadioInput == "OK :white_check_mark:":
        st.session_state.messages.append({"role": "user", "content": "SQL looks good :white_check_mark:", "type":"markdown"})
        st.session_state['sql']=st.session_state['tempSQL']
        st.session_state["df"] = vn.run_sql(sql=st.session_state['sql'])
        st.rerun() 
    else:
        st.stop()

elif st.session_state["df"] is not None and st.session_state["code"] is None:
    print('entering 3')    
    if st.session_state.get("show_table", True):
        df = st.session_state.get("df")
        # with st.chat_message("assistant"):
        if len(df) > 10:
            #st.text("First 10 rows of data")
            #st.dataframe(df.head(10))
            st.session_state.messages.append({"role": "assistant", "content": "First 10 rows of data"  , "type":"text"})
            st.session_state.messages.append({"role": "assistant", "content": df.head(10) , "type":"dataframe" })
        elif len(df) == 0:
            st.session_state.messages.append({"role": "assistant", "content": "Here are the results from the query:"  , "type":"text"})
            st.session_state.messages.append({"role": "assistant", "content": "Query returned zero rows, unable to make a figure please try again with a new question"  , "type":"text"})
            resetPrompt()
        else:
            st.session_state.messages.append({"role": "assistant", "content": df , "type":"dataframe" })
            #st.dataframe(df)
    # with user_message_sql_check:
    chart_instructions_input= "Please make the figure red in color and title it Nice Red figure"
        
        # if chart_instructions_input != '':
    st.session_state["code"] = vn.generate_plotly_code(question=st.session_state['prompt'], sql=st.session_state['sql'], df=df,chart_instructions=chart_instructions_input)
    if st.session_state.get("show_plotly_code", True):
        st.session_state.messages.append({"role": "assistant", "content": st.session_state["code"] , "type":"code" })
    st.rerun()

elif st.session_state["code"]  is not None and st.session_state.get("show_plotly_code", True):
    with st.chat_message("user"):
        plotyRadioInput = st.radio(
                "I would like to ...",
                options=["Edit :pencil2:", "OK :white_check_mark:"],
                index=None,
                captions = ["Edit the Plot code", "Use generated Plot code"],
                horizontal = True
            )
    if plotyRadioInput == "Edit :pencil2:":
        st.session_state.messages.append({"role": "user", "content": "I would like to Edit the Plot Code :pencil2:", 'type':"markdown" })
        st.warning("Please fix the generated Python code. Once you're done hit Shift + Enter to submit")
        
        python_code_response = code_editor(st.session_state["code"], lang="python")
        code = python_code_response["text"]
        
        st.session_state.messages.append({"role": "user", "content": code, 'type':'code' })
        st.rerun()
    elif plotyRadioInput == "K :white_check_mark:":
        st.session_state.messages.append({"role": "user", "content": "Plot code looks good! :white_check_mark:", 'type':'markdown'   })
        st.rerun()

elif st.session_state["code"]  is not None and st.session_state["fig"] is None:
    if st.session_state.get("show_chart", True):
        st.session_state["fig"] = vn.get_plotly_figure(plotly_code=st.session_state["code"] , df=st.session_state["df"])
        if st.session_state["fig"] is not None:
            st.session_state.messages.append({"role": "assistant", "content": st.session_state["fig"]  , 'type':'figure' })
        else:
            st.session_state.messages.append({"role": "assistant", "content":"I couldn't generate a chart" , 'type':'error' })
        st.rerun()
else:
    st.stop()


