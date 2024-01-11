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
import re

def is_select_statement(s):
    pattern = r"^\s*SELECT\s"
    return bool(re.match(pattern, s, re.IGNORECASE))


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


st.set_page_config(layout="wide")
st.title("Data G.E.N.I.E")
tab1,tab2 = st.tabs(['Chatbot',"🗃 SQL KnowledgeBase"])

tab2.subheader("Training Data")
tab2.dataframe(vn.get_training_data())

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help?" , "type":"markdown"}]
    st.session_state.prompt = None
    st.session_state.sql =None
    st.session_state.code =None
    st.session_state.df =None
    st.session_state.fig =None
    st.session_state.tempSQL =None
    st.session_state.tempCode =None
    st.session_state.enableUserTextInput =True
    st.session_state.saveQnAPair =None
    myQuestion = None

def resetPrompt():
    st.session_state['prompt'] = None
    st.session_state['sql'] =None
    st.session_state['code'] =None
    st.session_state['df'] =None
    st.session_state['fig'] =None
    st.session_state['tempSQL'] =None
    st.session_state['saveQnAPair'] =None
    st.session_state['tempCode'] =None
    st.session_state['enableUserTextInput'] =True
    myQuestion = None
    st.rerun()

def reRunClearApp():
    st.session_state['messages']= [{"role": "assistant", "content": "How can I help?" , "type":"markdown"}]
    st.session_state['prompt'] = None
    st.session_state['sql'] =None
    st.session_state['code'] =None
    st.session_state['df'] =None
    st.session_state['fig'] =None
    st.session_state['tempSQL'] =None
    st.session_state['saveQnAPair'] =None
    st.session_state['tempCode'] =None
    st.session_state['enableUserTextInput'] =True
    myQuestion = None

st.sidebar.title("Output Settings")
st.sidebar.checkbox("Show SQL", value=True, key="show_sql")
st.sidebar.checkbox("Show Table", value=True, key="show_table")
st.sidebar.checkbox("Show Plotly Code", value=True, key="show_plotly_code")
st.sidebar.checkbox("Show Chart", value=True, key="show_chart")
st.sidebar.checkbox("Show Follow-up Questions", value=True, key="show_followup")
st.sidebar.checkbox("Show Session State", value=True, key="show_sessionstate")
st.sidebar.button("Rerun", on_click=reRunClearApp, use_container_width=True)

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

if st.session_state.get('show_sessionstate',True):  
    st.sidebar.write(st.session_state)

    
if   myQuestion :=  st.chat_input( "Ask me a question about your data", disabled= not st.session_state['enableUserTextInput'] ) :

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
    st.session_state['enableUserTextInput']=False
    st.rerun()

elif st.session_state['prompt'] is not None and st.session_state['tempSQL'] is None:   
    print('entering 2') 
    st.session_state['tempSQL']= vn.generate_sql(question=st.session_state['prompt'])

    if is_select_statement(st.session_state['tempSQL']) == False:
        #responses is not a select statement let user ask again
        st.session_state.messages.append({"role": "assistant", "content": st.session_state['tempSQL'] , "type":"sql"})
        st.session_state['enableUserTextInput']=True
        st.session_state['prompt']=None
        st.session_state['tempSQL']=None
        st.rerun()

    if st.session_state.get("show_sql", True):
        st.session_state.messages.append({"role": "assistant", "content": st.session_state['tempSQL'] , "type":"sql"})
        st.rerun()
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
        st.warning("Please update the generated SQL code. Once you're done hit Shift + Enter to submit" )
        
        sql_response = code_editor(st.session_state['tempSQL'], lang="sql")
        fixed_sql_query = sql_response["text"]

        if fixed_sql_query != "":
            st.session_state.messages.append({"role": "user", "content": "I would like to edit the SQL :pencil2:" , "type":"markdown"  })
            # st.session_state.messages.append({"role": "user", "content": ":pencil: Edited SQL: ", "type":"markdown"})
            st.session_state['sql'] = "--Edited SQL:\n"+ fixed_sql_query
            st.session_state["df"] = vn.run_sql(sql=st.session_state['sql'])
            st.session_state.messages.append({"role": "user", "content":  st.session_state['sql'] , "type":"sql"})
            st.rerun()
        else:
            st.stop()
    elif sqlRadioInput == "OK :white_check_mark:":
        st.session_state.messages.append({"role": "user", "content": "SQL looks good :white_check_mark:", "type":"markdown"})
        st.session_state['sql']=st.session_state['tempSQL']
        st.session_state["df"] = vn.run_sql(sql=st.session_state['sql'])
        st.rerun() 
    else:
        st.stop()

elif st.session_state["df"] is not None and st.session_state["tempCode"] is None:
    print('entering 3')   
    df = st.session_state.get("df")
    if st.session_state.get("show_table", True):
        # with st.chat_message("assistant"):
        if len(df) > 10:
            #st.text("First 10 rows of data")
            #st.dataframe(df.head(10))
            st.session_state.messages.append({"role": "assistant", "content": "First 10 rows of data"  , "type":"markdown"})
            st.session_state.messages.append({"role": "assistant", "content": df.head(10) , "type":"dataframe" })
        elif len(df) == 0:
            st.session_state.messages.append({"role": "assistant", "content": "Here are the results from the query:"  , "type":"markdown"})
            st.session_state.messages.append({"role": "assistant", "content": df , "type":"dataframe" })
            st.session_state.messages.append({"role": "assistant", "content": "Query returned zero rows, unable to make a figure please try again with a new question"  , "type":"markdown"})
            resetPrompt()
        else:
            st.session_state.messages.append({"role": "assistant", "content": "Here are the results from the query:"  , "type":"markdown"})
            st.session_state.messages.append({"role": "assistant", "content": df , "type":"dataframe" })
            #st.dataframe(df)

    # with user_message_sql_check:
    chart_instructions_input= "Please make the figure red in color and title it Nice Red figure"
        
    # if chart_instructions_input != '':
    st.session_state["tempCode"] = vn.generate_plotly_code(question=st.session_state['prompt'], sql=st.session_state['sql'], df=df,chart_instructions=chart_instructions_input)

    if st.session_state.get("show_plotly_code", True):
        st.session_state.messages.append({"role": "assistant", "content": "Here is the code we can use to generate a figure from the data:"  , "type":"markdown"})
        st.session_state.messages.append({"role": "assistant", "content": st.session_state["tempCode"] , "type":"code" })
        st.rerun()
    else:
        st.session_state["code"] = st.session_state["tempCode"] 
        st.rerun()

elif st.session_state["tempCode"]  is not None and st.session_state["code"]  is  None :
    print('entering 4')   
    with st.chat_message("user"):
        plotyRadioInput = st.radio(
                "I would like to ...",
                options=["Edit :pencil2:", "OK :white_check_mark:"],
                index=None,
                captions = ["Edit the Plot code", "Use generated Plot code"],
                horizontal = True
            )
    if plotyRadioInput == "Edit :pencil2:":
        
        st.warning("Please fix the generated Python code. Once you're done hit Shift + Enter to submit")
        python_code_response = code_editor(st.session_state["tempCode"], lang="python")
        fixed_python_code = python_code_response["text"]

        if fixed_python_code != "":
            st.session_state.messages.append({"role": "user", "content": "I would like to Edit the Plot Code :pencil2:", 'type':"markdown" })
            st.session_state["code"] = "#Edited Python Code:\n"+python_code_response["text"]
            # st.session_state.messages.append({"role": "user", "content": ":pencil: Edited code: ", "type":"markdown"})
            st.session_state.messages.append({"role": "user", "content": st.session_state["code"], 'type':'code' })
            st.rerun()
        else:
            st.stop()
    elif plotyRadioInput == "OK :white_check_mark:":
        st.session_state.messages.append({"role": "user", "content": "Plot code looks good! :white_check_mark:", 'type':'markdown'   })
        st.session_state["code"] = st.session_state["tempCode"]
        st.rerun()

elif st.session_state["code"] is not None and st.session_state["fig"] is None:
    print('entering 5')
    if st.session_state.get("show_chart", True):
        st.session_state["fig"] = vn.get_plotly_figure(plotly_code=st.session_state["code"] , df=st.session_state["df"])
        if st.session_state["fig"] is not None:
            st.session_state.messages.append({"role": "assistant", "content": "Here is the generated figure:"  , "type":"markdown"})
            st.session_state.messages.append({"role": "assistant", "content": st.session_state["fig"]  , 'type':'figure' })
            st.session_state.messages.append({"role": "assistant", "content": "Do you want to save this question and SQL answer-pair to my knowledge base?"  , "type":"markdown"})

        else:
            st.session_state.messages.append({"role": "assistant", "content":"I couldn't generate a chart" , 'type':'error' })
        st.rerun()

elif st.session_state["fig"] is not None and st.session_state["saveQnAPair"] is None:

    print('entering 6')   
    with st.chat_message("user"):
        plotyRadioInput = st.radio(
                "I would like to ...",
                options=["Yes :floppy_disk:", "No :x:"],
                index=None,
                captions = ["Yes, Save the question and SQL answer-pair", "No, do not save"],
                horizontal = True
            )
    if plotyRadioInput == "Yes :floppy_disk:":
        st.session_state.messages.append({"role": "user", "content": "Yes, Save the question and SQL answer-pair :floppy_disk:", 'type':"markdown" })
        vn.add_question_sql(st.session_state["prompt"],st.session_state["sql"])
        st.session_state.messages.append({"role": "assistant", "content": "Done!"  , "type":"markdown"})
        st.session_state["saveQnAPair"] = True
        st.rerun()
    elif plotyRadioInput == "No :x:":
        st.session_state.messages.append({"role": "assistant", "content": "No, do not save the question and SQL answer-pair :x:", 'type':'markdown'   })
        st.session_state["saveQnAPair"] = False
        st.rerun()
elif st.session_state["fig"] is not None and st.session_state["saveQnAPair"] == False:
    st.session_state.messages.append({"role": "assistant", "content": "Got it, let start over by. Please go ahead and ask a new question", 'type':'markdown'   })
    resetPrompt()
    
#TODO: provide prompoting questions 
#TODO: Restart question of as
#TODO: save the conversation 
#TODO: export coversation as pdf, email 
else:
    st.stop()