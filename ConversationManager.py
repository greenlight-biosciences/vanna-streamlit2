import time
from code_editor import code_editor
#from utils.vanna_calls import *
import streamlit as st
#import vanna as vn
# import os
import streamlit as st
# from bokeh.plotting import figure

from myVanna  import *
import os
from dotenv import load_dotenv
load_dotenv(".env")


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



def trainVN(input , type, question =None):
    if type =='ddl':
        vn.train(ddl=input)
    elif type =='doc':
        vn.train(documentation=input)
    elif type =='sql':
         # Check if question is provided
        if question:
            vn.train(sql=input, question=question)
        else:
            vn.train(sql=input)   

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
    st.session_state['ddl'] =None
    st.session_state['doc'] =None
    st.session_state['sqlQ'] =None
    st.session_state['sqlA'] =None
    st.session_state['figureInstructions'] =None
    st.session_state['userUpdateCode']=None
    userResponse = None
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
    st.session_state['ddl'] =None
    st.session_state['doc'] =None
    st.session_state['sqlQ'] =None
    st.session_state['sqlA'] =None
    st.session_state['figureInstructions'] =None
    st.session_state['userUpdateCode']=None
    userResponse = None

def trainQuestionAnswer(sqlQ=None,sqlA=None):
    if(sqlA and sqlQ):
        trainVN(input =sqlQ , question=sqlA, type ='sql')
        st.session_state.sqlQ_input = ""
        st.session_state.sqlA_input = ""
def trainDoc(doc):
    if (doc):
        trainVN(input =doc, type ='doc')
        st.session_state.doc_input = ""

def trainDDL(ddl):
    if (ddl):
        trainVN(input =ddl, type ='ddl')
        st.session_state.ddl_input = ""

def runTrainingPlan(self, type):
        if type =='Snowflake':
            self.get_training_plan_generic()
        else:
            self.get_training_plan_generic()

from myVanna import MyVanna
from utility import *
# from init_app_vars import *
# from typing import Union


st.set_page_config(layout="wide")
st.title("Data G.E.N.I.E")
tab1,tab2 = st.tabs(['Chatbot',"🗃 SQL KnowledgeBase"])


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
    st.session_state.ddl =None
    st.session_state.doc =None
    st.session_state.sqlQ =None
    st.session_state.sqlA =None
    st.session_state.enableTraining=False
    st.session_state.figureInstructions=None
    st.session_state.userUpdateCode=None
    st.session_state.plottingLib='Plotly'
    userResponse = None

def deleteTraining():
    selected_rows = selectedForDeletion[selectedForDeletion['Select']]
    selected_values = selected_rows['id']

    # Iterate over selected IDs and remove training data
    for id_value in selected_values:
        try:
            vn.remove_training_data(id_value)
        except Exception as e:
            print(f"Error removing training data for id {id_value}: {e}")



tab2.subheader("Training Data")
tab2.button('Delete Training',key='deleteTraining', on_click=deleteTraining)
tab2.button('Run Automated DB schema Training',key='autoTraining', on_click=runTrainingPlan, args=(os.environ.get("DATABASETYPE"),))

trainingData= vn.get_training_data()
trainingData.insert(0, "Select", False)

selectedForDeletion = tab2.data_editor(trainingData, hide_index=True, column_config={"Select": st.column_config.CheckboxColumn(required=True), "id":st.column_config.Column(width='small')},)

st.session_state['ddl']=tab2.text_area('Enter DDL information', value='', height=None, max_chars=None, key='ddl_input',disabled=st.session_state['enableTraining'] , on_change=trainDDL, args=(st.session_state['ddl'], ))
st.session_state['doc']=tab2.text_area('Enter Documentation information:', value='', height=None, max_chars=None, key='doc_input',disabled=st.session_state['enableTraining'], on_change=trainDoc, args=(st.session_state['doc'], ))
st.session_state['sqlQ']= tab2.text_area('Enter Question:', value='', height=None, max_chars=None, key='sqlQ_input',disabled=st.session_state['enableTraining'], on_change=trainQuestionAnswer, args=(st.session_state['sqlQ'],st.session_state['sqlA'],  ))
st.session_state['sqlA']= tab2.text_area('Enter SQL Answer:', value='', height=None, max_chars=None, key='sqlA_input',disabled=st.session_state['enableTraining'], on_change=trainQuestionAnswer, args=(st.session_state['sqlQ'],st.session_state['sqlA'],  ))



st.sidebar.title("Output Settings")
st.sidebar.checkbox("Show SQL", value=True, key="show_sql")
st.sidebar.checkbox("Show Table", value=True, key="show_table")
st.sidebar.checkbox("Show Plotly Code", value=True, key="show_plotly_code")
st.sidebar.checkbox("Show Chart", value=True, key="show_chart")
st.sidebar.checkbox("Show Follow-up Questions", value=True, key="show_followup")
st.sidebar.checkbox("Show Session State", value=True, key="show_sessionstate")
st.sidebar.button("Rerun", on_click=reRunClearApp, use_container_width=True)
st.session_state['plottingLib']=st.sidebar.selectbox('Plotting Library',options=['Plotly','Altair','Bokeh'],index=0)

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
        elif message["type"] =='figure-code':
            col1, col2 = st.columns([3, 2], )
            col1.markdown('Here is a figure for the data:')
            if st.session_state['plottingLib']=='Plotly':
                col1.plotly_chart(message["figure"])
            elif  st.session_state['plottingLib']=='Altair':
                col1.altair_chart(message["figure"])
            elif st.session_state['plottingLib']=='Bokeh':
                col1.bokeh_chart(message["figure"])
            else:
                st.text(message["content"])
            col2.markdown('Here is the code for the figure:')
            col2.code(message["code"], language="python", line_numbers=True )
        elif message["type"] =='figure':
            st.plotly_chart(message["content"])
        elif  message["type"] =='error':
            st.error(message["content"])
        else:
            st.text(message["content"])

if st.session_state.get('show_sessionstate',True):  
    st.sidebar.write(st.session_state)

    
if   userResponse :=  st.chat_input( "Ask me a question about your data", disabled= not st.session_state['enableUserTextInput'] ) :

    print('entering 1')
    # with st.chat_message("assistant"):and st.session_state['enableUserTextInput'] is True 
    #     questions = vn.generate_questions() 

    #     for i, question in enumerate(questions):
    #         #time.sleep(0.05)
    #         button = st.button(
    #             question,
    #             on_click=set_question,
    #             args=(question,),
    #         )
    #     st.session_state.messages.append({"role": "assistant", "content": "\n-".join(questions),'type': 'markdown'})
    if(st.session_state['prompt'] is  None):
        st.session_state.messages.append({"role": "user", "content": userResponse, "type":"markdown"})
        st.session_state['prompt'] =userResponse
        st.session_state['enableUserTextInput']=False
        st.rerun()
    elif(st.session_state['figureInstructions'] is  None):
        st.session_state['figureInstructions'] =userResponse
        st.session_state['enableUserTextInput']=False
        st.rerun()


elif st.session_state['prompt'] is not None and st.session_state['tempSQL'] is None:   
    print('entering 2') 
    st.session_state['tempSQL']= vn.generate_sql(question=st.session_state['prompt'])

    if is_select_statement(st.session_state['tempSQL']) == False:
        #responses is not a select statement let user ask again
        st.session_state.messages.append({"role": "assistant", "content": st.session_state['tempSQL'] , "type":"markdown"})
        st.session_state['enableUserTextInput']=True
        st.session_state['prompt']=None
        st.session_state['tempSQL']=None
        st.rerun()
    elif st.session_state.get("show_sql", True):
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

    if st.session_state['figureInstructions'] is None:
        st.session_state["tempCode"] = vn.generate_plotly_code(question=st.session_state['prompt'], sql=st.session_state['sql'], df=df,chart_instructions='', plottingLib=st.session_state['plottingLib'] )
    else:
        st.session_state["tempCode"] = vn.edit_plotly_code(question=st.session_state['prompt'], sql=st.session_state['sql'], df=st.session_state["df"],chart_instructions=st.session_state["figureInstructions"],chart_code=st.session_state["userUpdateCode"],plottingLib=st.session_state['plottingLib']  )
        st.session_state['figureInstructions'] = None

    st.session_state["fig"] = vn.get_plotly_figure(plotly_code=st.session_state["tempCode"] , df=st.session_state["df"])

    if st.session_state.get("show_plotly_code", True):
        st.session_state.messages.append({"role": "assistant", "figure": st.session_state["fig"] ,'code':st.session_state["tempCode"], "type":"figure-code" })
        st.rerun()
    elif st.session_state.get("show_chart", True):
        if st.session_state["fig"] is not None:
            st.session_state.messages.append({"role": "assistant", "content": f"For your question: {st.session_state['prompt']} - we prepared this final figure:"  , "type":"markdown"})
            st.session_state.messages.append({"role": "assistant", "content": st.session_state["fig"]  , 'type':'figure' })
        else:
            st.session_state.messages.append({"role": "assistant", "content":"I couldn't generate a chart" , 'type':'error' })
        st.rerun()

elif st.session_state["tempCode"]  is not None and st.session_state["code"]  is  None  and st.session_state["figureInstructions"] is None and st.session_state['enableUserTextInput'] == False:
    print('entering 4') 
    with st.chat_message("user"):
        plotyRadioInput = st.radio(
                "I would like to ...",
                options=["Instruct Changes","Edit Code Manually :pencil2:", "OK :white_check_mark:"],
                index=None,
                captions = ["Tell the Chat Bot how the figure should be updated","Edit the Plot code", "Use generated Plot code"],
                horizontal = True
            )
    if plotyRadioInput == "Edit Code Manually :pencil2:":
        
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
        st.session_state.messages.append({"role": "assistant", "content": "Do you want to save this question and SQL answer-pair to my knowledge base?"  , "type":"markdown"})
        st.session_state["code"] = st.session_state["tempCode"]
        st.rerun()
    elif plotyRadioInput == "Instruct Changes":
        print('entering Instruct Changes')   
        st.session_state['enableUserTextInput'] = True
        st.session_state["fig"] = None
        st.rerun()
    else:
        st.stop()
        
elif st.session_state["tempCode"]  is not None and st.session_state["code"]  is  None  and st.session_state["figureInstructions"] is not None:
    print('entering 4.5') 
    st.session_state.messages.append({"role": "user", "content": st.session_state["figureInstructions"],  'type':'markdown'   })
    st.session_state["userUpdateCode"] = st.session_state["tempCode"] 
    # st.session_state["figureInstructions"] = None
    st.session_state["tempCode"] = None
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
        st.session_state.messages.append({"role": "user", "content": "No, do not save the question and SQL answer-pair :x:", 'type':'markdown'   })
        st.session_state["saveQnAPair"] = False
        st.rerun()
    else:
        st.stop()
elif st.session_state["fig"] is not None and st.session_state["saveQnAPair"] is not None:
    st.session_state.messages.append({"role": "assistant", "content": "Got it, Im ready for your next question. Please go ahead and ask a new question...", 'type':'markdown'   })
    resetPrompt()
    
#TODO: provide prompoting questions 
#TODO: Restart question of as
#TODO: save the conversation 
#TODO: export coversation as pdf, email 
else:
    st.stop()