import time
from code_editor import code_editor
import streamlit as st
import streamlit as st
from datetime import datetime
from utility import *
from myVanna  import *
import os
from dotenv import load_dotenv
from streamlit_modal import Modal
from streamlit.web.server.websocket_headers import _get_websocket_headers


vn= MyVanna(
    config={	
        'api_type': 'azure',
        'api_base': os.environ.get("AZURE_OPENAI_ENDPOINT"),
	    'api_version': '2023-05-15',
	    #'engine': os.environ.get("AZUREOPENAIENGINE"),
        'model': os.environ.get("AZUREOPENAIENGINE"),
	    'api_key': os.environ.get("AZUREOPENAIKEY"),
        'authtype': os.environ.get("CHROMASDBAUTHTYPE"),
        'path':os.environ.get("FILE_SHARE_PATH","."),
})

appTitle = os.environ.get("APPTITLE")
menu_items ={"Get help":os.environ.get("GETHELPURL"), "Report a Bug":os.environ.get("SUBMITTICKETURL") }
st.set_page_config(layout="wide", page_title =appTitle, menu_items =menu_items )
st.title(appTitle)


def getUserID(): 
    headers = _get_websocket_headers()
    user_email ='testuser@greenlightbio.com'
    if "X-Ms-Client-Principal-Name" in headers:
        user_email = headers["X-Ms-Client-Principal-Name"]
        vn.logInfo(f'Logged in user: {user_email}')
    return user_email.split('@')[0]

# Initialize app state variables
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
    st.session_state.vnModel='PHGENERAL'
    st.session_state.sqlInstructions =None
    st.session_state.userUpdateSQL =None
    st.session_state.textInputHelp =None
    st.session_state.uniqWidgetCounter =0
    st.session_state.sqlRadioInput = None
    st.session_state.enablePlottingDataModelChange = True
    st.session_state.UserID= None
    userResponse = None

st.session_state['UserID'] = getUserID()
st.markdown( f"##### Welcome : {st.session_state['UserID']}")

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
    st.session_state['sqlInstructions'] =None
    st.session_state['userUpdateSQL'] =None
    st.session_state['textInputHelp'] =None
    st.session_state['enablePlottingDataModelChange'] =True
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
    st.session_state['sqlInstructions'] =None
    st.session_state['userUpdateSQL'] =None
    st.session_state['textInputHelp'] =None
    st.session_state['uniqWidgetCounter'] =0
    st.session_state['sqlRadioInput'] =None
    st.session_state['enablePlottingDataModelChange'] =True
    userResponse = None

def trainQuestionAnswer(sqlQ=None,sqlA=None):
    vn.logInfo('running trainQuestionAnswer traning')

    if(sqlA and sqlQ):
        returnVal= vn.trainVN(input = sqlA, question=sqlQ, type ='sql' , schema= st.session_state['vnModel'] )
        if returnVal:
            st.session_state.sqlQ_input = ""
            st.session_state.sqlA_input = ""
            st.toast(f'Added Question & SQL Answer to Knowledgebase for Schema:{st.session_state["vnModel"]}')
        else:
            st.toast('Failed to add Question & SQL Answer to Knowledgebase')
    else:
        st.toast('No Question & SQL Answer entered!')



def trainDoc(doc):
    vn.logInfo('running doc traning')
    if (doc):
        returnVal = vn.trainVN(input =doc, type ='doc', schema= st.session_state['vnModel'])
        if returnVal:
            st.session_state.doc_input = ""
            st.toast(f'Added Documenation to Knowledgebase for Schema:{st.session_state["vnModel"]}')
        else:
            st.toast('Failed to add Documenation to Knowledgebase')
    else:
        st.toast('No Documenation entered!')


def trainDDL(ddl):
    vn.logInfo('running ddl traning')

    if (ddl):
        returnVal = vn.trainVN(input =ddl, type ='ddl', schema= st.session_state['vnModel'])
        if returnVal:
            st.session_state.ddl_input = ""
            st.toast(f'Added DDL to Knowledgebase for Schema:{st.session_state["vnModel"]}')
        else:
            st.toast('Failed to add DDL to Knowledgebase')
    else:
        st.toast('No DDL entered!')

def runAutomatedTrainingPlan(schema:str=None, database:str='PH_DATALAKE_DEV'):
    vn.runTrainingPlanSnowflake(schema=schema, database=database)

tab1,tab2,tab3 = st.tabs(['Chatbot',"🗃 SQL KnowledgeBase", 'Introduction Guide'])

def deleteTraining():
    selected_rows = selectedForDeletion[selectedForDeletion['Select']]
    selected_values = selected_rows['id']

    # Iterate over selected IDs and remove training data
    for id_value in selected_values:
        try:
            vn.remove_training_data(id_value)
        except Exception as e:
            vn.logError(f"Error removing training data for id {id_value}: {e}")

@st.cache_data 
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


def generate_fileName():
    # Get the current time and format it as a string
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Use the formatted time string in the filename
    return f'{current_time}-DataGenieExport.csv'


def generate_uniqObjectName(object:str='obj'):
    # Get the current time and format it as a string
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state["uniqWidgetCounter"] = st.session_state["uniqWidgetCounter"] +1
    # Use the formatted time string in the filename
    return f'{current_time}-{object}-{st.session_state["uniqWidgetCounter"]}'

# @st.cache_resource
    # vn.connect_to_snowflake(
    #     account=os.environ.get('ACCOUNT'),
    #     username=os.environ.get('SNOWFLAKE_USER'),
    #     password=os.environ.get('SNOWFLAKE_PASS'),
    #     database=os.environ.get('SNOWFLAKE_DATABASE'),
    #     role=os.environ.get('ROLE'),
    #     schema=model,
    #     warehouse=os.environ.get('WAREHOUSE'))
    



st.sidebar.title("Output Settings")

def changeSchemaCallback():
    vn.logInfo(f"Schema Changed to: {st.session_state['vnModel']}")


def process_file(uploaded_file):
    # Read the file with pandas
    data = pd.read_csv(uploaded_file)
    # Initialize progress bar
    progress_bar = st.progress(0)
    total_rows = len(data)
    processed_count = 0  # To keep track of the number of processed rows

    for index, row in data.iterrows():
        # Update the progress bar
        progress_bar.progress(index / total_rows)
        if row['Select']:  # Only process selected rows
            # Check for empty content
            if pd.notna(row['content']):
                if row['training_data_type'].lower() == 'sql':
                    # Process SQL question-answer pair
                    vn.trainVN(input = row['content'], question=row['question'], type ='sql' , schema= st.session_state['vnModel'] )
                elif row['training_data_type'].lower() == 'doc':
                    # Process documentation
                    vn.trainVN(input =['content'], type ='doc', schema= st.session_state['vnModel'])
                elif row['training_data_type'].lower() == 'ddl':
                    # Process DDL
                    vn.trainVN(input =row['content'], type ='ddl', schema= st.session_state['vnModel'])
                processed_count += 1
            else:
                st.warning(f"Row {index} has empty content and will be skipped.")
        else:
            st.info(f"Row {index} not selected for processing.")
    
    # Finalize the progress bar
    progress_bar.progress(100)
    st.success(f'Processed {processed_count} out of {total_rows} rows.')

st.sidebar.checkbox("Show SQL", value=True, key="show_sql", disabled=True)
st.sidebar.checkbox("Show Table", value=True, key="show_table", disabled=True)
st.sidebar.checkbox("Show Plotly Code", value=True, key="show_plotly_code", disabled=True)
st.sidebar.checkbox("Show Chart", value=True, key="show_chart", disabled=True)
st.sidebar.checkbox("Show Follow-up Questions", value=False, key="show_followup", disabled=True)
st.sidebar.checkbox("Show Session State", value=False, key="show_sessionstate")
st.sidebar.button("Reset/Clear Conversation", on_click=reRunClearApp, use_container_width=True)
st.session_state['plottingLib']=st.sidebar.selectbox('Plotting Library',options=['Plotly','Altair','Bokeh'],index=0, disabled= not st.session_state['enablePlottingDataModelChange'], help='Change which plotting library chat app uses for generating figures (Note:Changing the plotting library is only allowed at the start of a new conversation)')
vnModelValue=st.sidebar.selectbox('Data Mart',options=os.environ.get('ALLOWEDSCHEMAS','MISSING_ALLOWEDSCHEMAS_ENV').split(','),index=0, 
                                                on_change=changeSchemaCallback,
                                                #on_change=vn.connect_to_snowflake(),
                                                #args=(st.session_state['vnModel'],), 
                                                key='vnModel',
                                                        disabled= not st.session_state['enablePlottingDataModelChange'], help='Change which Data Mart the chat app is talking to (Note: Changing the Data Mart is only allowed at the start of a new conversation)')
st.sidebar.markdown( f"### User question:\n{st.session_state['prompt']}")
vn.logInfo(f"Confirming Schema Changed to: {vnModelValue}")


vn.connect_to_snowflake(
        account=os.environ.get('ACCOUNT'),
        username=os.environ.get('SNOWFLAKE_USER'),
        password=os.environ.get('SNOWFLAKE_PASS'),
        database=os.environ.get('SNOWFLAKE_DATABASE'),
        role=os.environ.get('ROLE'),
        schema=st.session_state['vnModel'],
        warehouse=os.environ.get('WAREHOUSE')
    )

tab2.subheader("Training Data")
tab2.button('Delete Training',key='deleteTraining', on_click=deleteTraining)
tab2.button('Run Automated DB schema Training',key='autoTraining', on_click= runAutomatedTrainingPlan, args=(st.session_state['vnModel'],os.environ.get('DATABASE'),))

trainingData= vn.get_training_data(schema=st.session_state['vnModel'])
trainingData.insert(0, "Select", False)

selectedForDeletion = tab2.data_editor(trainingData, hide_index=True, column_config={"Select": st.column_config.CheckboxColumn(required=True), "id":st.column_config.Column(width='small')},)
st.session_state['ddl']=tab2.text_area('Enter DDL information', value='', height=None, max_chars=None, key='ddl_input',disabled=st.session_state['enableTraining'] )
tab2.button('Submit DDL',key='submit_DDL', on_click=trainDDL, args=(st.session_state['ddl'], ))
tab2.markdown("***")
st.session_state['doc']=tab2.text_area('Enter Documentation information:', value='', height=None, max_chars=None, key='doc_input',disabled=st.session_state['enableTraining'])
tab2.button('Submit Docs',key='submit_DOC', on_click=trainDoc, args=(st.session_state['doc'], ))
tab2.markdown("***")
st.session_state['sqlQ']= tab2.text_area('Enter Question:', value='', height=None, max_chars=None, key='sqlQ_input',disabled=st.session_state['enableTraining'])
st.session_state['sqlA']= tab2.text_area('Enter SQL Answer:', value='', height=None, max_chars=None, key='sqlA_input',disabled=st.session_state['enableTraining'])
tab2.button('Submit Q&A',key='submit_Q&A', on_click=trainQuestionAnswer, args=(st.session_state['sqlQ'],st.session_state['sqlA'],  ))
tab2.markdown("***")
uploaded_file = tab2.file_uploader("Traing From Backup File - Choose a CSV file", type="csv")
if uploaded_file is not None:
    process_file(uploaded_file)

@st.cache_data
def cache_describeSQLData(prompt: str = None, sql:str =None, additionalInstructions:str =None, df_describe:str =None):
    return vn.describeSQLData(df_describe=df_describe, sql=sql, question=prompt,additionalInstructions=additionalInstructions)


for message in st.session_state.messages:
    try:
        with tab1.chat_message(message["role"]):
            if message["type"] =='markdown':
                st.markdown(message["content"])
            elif message["type"] =='code':
                st.code(message["content"], language="python", line_numbers=True )
            elif message["type"] =='sql':
                st.code(message["sql"], language="sql", line_numbers=True)
            # elif message["type"] =='dataframe-preview':
            #     st.markdown('Data Preview (first 5 rows):')
            #     st.dataframe(message["df"])
            #     st.download_button(
            #         key=generate_uniqObjectName('df-download2'),
            #         label=":floppy_disk: Download All Data as CSV",
            #         data=convert_df(message["df"]),
            #         file_name=generate_fileName(),
            #         mime='text/csv', disabled=message["df"].empty) 
            elif message["type"] =='dataframe':
                st.markdown(f'Data (first {message["nrows"]} rows):')
                st.dataframe(message["df"].head(int(message["nrows"])))
                st.download_button(
                    key=generate_uniqObjectName('df-download2'),
                    label=":floppy_disk: Download All Data as CSV",
                    data=convert_df(message["df"]),
                    file_name=generate_fileName(),
                    mime='text/csv',disabled=message["df"].empty) 
            elif message["type"] =='sql-dataframe':
                col1, col2 ,col3= st.columns([2, 2,1], )
                col1.markdown('Here is a the generated SQL query for your question:')
                col1.code(message["sql"], language="sql", line_numbers=True)
                col2.markdown(f"Data Preview (first {message['nrows']} rows):")
                col2.dataframe(message["df"].head(int(message["nrows"])))
                col2.download_button(
                    key=generate_uniqObjectName('df-download2'),
                    label=":floppy_disk: Download All Data as CSV",
                    data=convert_df(message["df"]),
                    file_name=generate_fileName(),
                    mime='text/csv',disabled=message["df"].empty)
                
                dataViewModal = Modal(
                        "Inspect Data", 
                        key="viewdata-modal",
                        
                        # Optional
                        padding=20,    # default value
                        max_width=744  # default value,
                    )
                if(col2.button('Inspect Data :mag:',key=generate_uniqObjectName('btInspect-Data'),disabled=message["df"].empty  )):
                    dataViewModal.open()
                    
                if(dataViewModal.is_open()):
                    with dataViewModal.container():
                        st.dataframe(message["df"])
                col3.markdown("***Quick Data Overview:***")
                col3.markdown(cache_describeSQLData(df_describe=message["df"].describe(), sql=message["sql"], prompt=message["prompt"]))
            elif message["type"] =='figure-code':
                col1, col2 = st.columns([3, 2], )
                col1.markdown('Here is a figure for the data:')
                if message['figtype']=='Plotly':
                    col1.plotly_chart(message["fig"],use_container_width=True)
                elif   message['figtype']=='Altair':
                    col1.altair_chart(message["fig"],use_container_width=True)
                elif  message['figtype']=='Bokeh':
                    col1.bokeh_chart(message["fig"],use_container_width=True)
                else:
                    st.text(message["content"])
                col2.markdown('Here is the code for the figure:')
                col2.code(message["code"], language="python", line_numbers=True )
            elif message["type"] =='figure':
                if message['figtype']=='Plotly':
                    col1.plotly_chart(message["fig"],use_container_width=True)
                elif  message['figtype']=='Altair':
                    col1.altair_chart(message["fig"],use_container_width=True)
                elif message['figtype']=='Bokeh':
                    col1.bokeh_chart(message["fig"],use_container_width=True)
                else:
                    st.text(message["content"])
            elif  message["type"] =='error':
                st.error(message["content"])
            else:
                st.text(message["content"])
    except Exception as e:
        vn.logError(e)
        st.error(f"Error - Failed to render chat message:\n {e}")
if st.session_state.get('show_sessionstate',True):  
    st.sidebar.write(st.session_state)

if userResponse :=  st.chat_input( optionSelector(st.session_state['textInputHelp']), disabled= not st.session_state['enableUserTextInput'] ) :

    vn.logInfo(f'Working with user input:{userResponse}')
   
    if(st.session_state['prompt'] is  None):
        st.session_state.messages.append({"role": "user", "content": userResponse, "type":"markdown"})
        st.session_state['prompt'] =userResponse
        st.session_state['enableUserTextInput']=False
        st.session_state['enablePlottingDataModelChange'] = False
        vn.logInfo(f'Set user input as Prompt:{userResponse}')
        st.rerun()
    elif(st.session_state['prompt'] is not None and st.session_state['tempSQL'] is None):
        st.session_state['enableUserTextInput']=False
        st.session_state.messages.append({"role": "user", "content": userResponse,  'type':'markdown'   })
        st.session_state['prompt']=vn.summarizePrompt(st.session_state['prompt'],questionConversationHistory= st.session_state['messages'])
        st.toast(f":sparkles: Your overall question has been updated, see side bar : {st.session_state['prompt']}" )
        st.rerun()
    elif(st.session_state['figureInstructions'] is None and st.session_state["tempCode"] is not None):
        st.session_state['figureInstructions'] =userResponse
        st.session_state["userUpdateCode"] = st.session_state["tempCode"] 
        st.session_state['enableUserTextInput']=False
        st.session_state["tempCode"] =None
        st.session_state.messages.append({"role": "user", "content": st.session_state["figureInstructions"],  'type':'markdown'   })
        st.rerun()
elif st.session_state['prompt'] is not None and st.session_state['tempSQL'] is None and st.session_state['enableUserTextInput']==False:   
    vn.logInfo('Generating SQL for user prompt') 

    st.session_state['tempSQL']= vn.generate_sql(question=st.session_state['prompt'], questionConversationHistory=st.session_state['messages'], schema=st.session_state['vnModel'])

    if detect_sql_statement_at_line_start(st.session_state['tempSQL']) == False:
        #responses is not a select statement let user ask again
        st.session_state.messages.append({"role": "assistant", "content": st.session_state['tempSQL'] , "type":"markdown"})
        st.session_state['enableUserTextInput']=True
        st.session_state['prompt']=None
        st.session_state['tempSQL']=None
        st.rerun()
    else:
        try:
            st.session_state["df"] = vn.run_sql(sql=st.session_state['tempSQL'])
            if st.session_state.get("show_sql", True):
                st.session_state.messages.append({"role": "assistant", "sql": st.session_state['tempSQL'] , 'df':st.session_state['df'], 'prompt':st.session_state['prompt'], 'nrows':'5',"type":"sql-dataframe"})
                st.rerun()
            else:
                st.session_state.messages.append({"role": "assistant", "content": 'I was able to successfully write a query for your question' , "type":"markdown"})
                st.rerun()
        except Exception as e:
            vn.logError(e)
            st.session_state.messages.append({"role": "assistant", "sql": st.session_state['tempSQL'] ,"type":"sql"})
            st.session_state.messages.append({"role": "assistant", 'content': f'There was an error running the generated SQL above:\n{e}', 'type':'error'})
            st.rerun()
elif st.session_state['tempSQL'] is not None and st.session_state['sql'] is None and st.session_state['sqlInstructions'] is None and st.session_state['enableUserTextInput'] == False :
    vn.logInfo('Checking with user on next steps after generating SQL') 
    
    with st.chat_message("user"):
        st.session_state['sqlRadioInput'] = st.radio(
            "I would like to ...",
            options=["Instruct Changes (SQL):speaking_head_in_silhouette:","Edit :pencil2:", "OK :white_check_mark:","Restart with a New Question :wastebasket:"],
            index=None,
            captions = ["Tell Data GENIE how the SQL query should be updated","Edit the SQL", "Use generated SQL query", "Clear current question and start over"],
            horizontal = True,
        # key='sqlRadioInput'
        )
    st.session_state['textInputHelp'] =st.session_state['sqlRadioInput']
    if st.session_state['sqlRadioInput'] == "Edit :pencil2:":
        st.warning("Please update the generated SQL code. Once you're done hit Ctrl + Enter to submit" )
        
        sql_response = code_editor(st.session_state['tempSQL'], lang="sql")
        fixed_sql_query = sql_response["text"]

        if fixed_sql_query != "":
            st.session_state.messages.append({"role": "user", "content": "I would like to edit the SQL :pencil2:" , "type":"markdown"  })

            st.session_state['tempSQL'] = fixed_sql_query
            try:
                df = vn.run_sql(sql=st.session_state['tempSQL'])
                st.session_state.messages.append({"role": "user", "sql":  "--Edited SQL:\n"+st.session_state['tempSQL'] , "type":"sql"})
                st.session_state.messages.append({"role": "assistant", 'content':"Result for your edited SQL query","type":"markdown"})
                st.session_state.messages.append({"role": "assistant", 'df':df, 'nrows':'5',"type":"dataframe"})
                st.session_state['sqlRadioInput']=None
            except Exception as e:
                vn.logError(e)
                st.session_state.messages.append({"role": "assistant", "sql": st.session_state['tempSQL'] ,"type":"sql"})
                st.session_state.messages.append({"role": "assistant", 'content': f'There was an error running the generated SQL above:\n{e}', 'type':'error'})
                st.rerun()
        else:
            st.stop()
    elif st.session_state['sqlRadioInput'] == "OK :white_check_mark:":
        st.session_state.messages.append({"role": "user", "content": "SQL looks good :white_check_mark:", "type":"markdown"})
        st.session_state['sql']=st.session_state['tempSQL']
        st.session_state["df"] = vn.run_sql(sql=st.session_state['sql'])
        st.rerun() 
    elif st.session_state['sqlRadioInput'] == "Instruct Changes (SQL):speaking_head_in_silhouette:":
        vn.logInfo(f'User requests to Instruct SQL Changes for sql:{st.session_state["tempSQL"]}')   
        st.session_state['enableUserTextInput'] = True
        st.session_state["tempSQL"] = None
        st.session_state["df"] = None
        st.rerun()
    elif st.session_state['sqlRadioInput'] == "Restart with a New Question :wastebasket:":
        st.session_state.messages.append({"role": "user", "content": "Lets restart with a new question"  , "type":"markdown"})
        st.session_state.messages.append({"role": "assistant", "content": "Got it, Im ready for your next question. Please go ahead and ask a new question...", 'type':'markdown'   })
        resetPrompt()
    else:
        st.stop()

elif st.session_state["df"] is not None and st.session_state["tempCode"] is None  and st.session_state["enableUserTextInput"] == False :
    vn.logInfo('Working on generating a Figure for users prompt')   
    df = st.session_state.get("df")
    if st.session_state.get("show_table", True) and st.session_state.get("userUpdateCode") is None:
        if len(df) > 10:
            st.session_state.messages.append({"role": "assistant", "content": "Here are the results from the query (first 10 rows):"  , "type":"markdown"})
            st.session_state.messages.append({"role": "assistant", "df": df ,"nrows": 10, "type":"dataframe" })
        elif len(df) == 0:
            st.session_state.messages.append({"role": "assistant", "content": "Here are the results from the query:"  , "type":"markdown"})
            st.session_state.messages.append({"role": "assistant", "df": df ,"nrows": '0', "type":"dataframe" })
            st.session_state.messages.append({"role": "assistant", "content": "Query returned zero rows, unable to make a figure please try again with a new question"  , "type":"markdown"})
            resetPrompt()
        else:
            st.session_state.messages.append({"role": "assistant", "content": "Here are the results from the query:"  , "type":"markdown"})
            st.session_state.messages.append({"role": "assistant", "df": df ,"nrows": len(df), "type":"dataframe" })
    

    if st.session_state['figureInstructions'] is None:
        st.session_state["tempCode"] = vn.generate_plot_code(question=st.session_state['prompt'], sql=st.session_state['sql'], df_metadata=df.dtypes,df=df ,chart_instructions='', plottingLib=st.session_state['plottingLib'] )
    else:
        st.session_state["tempCode"] = vn.edit_plot_code(question=st.session_state['prompt'], sql=st.session_state['sql'], df_metadata=df.dtypes,df=df,chart_instructions=st.session_state["figureInstructions"],chart_code=st.session_state["userUpdateCode"],plottingLib=st.session_state['plottingLib']  )
        st.session_state['figureInstructions'] = None
    vn.logInfo(f'Plot code generated {st.session_state["tempCode"]}')
    plottingError =None
    if st.session_state.get("show_plotly_code", True):
        if st.session_state['plottingLib'] == 'Plotly':
            st.session_state["fig"], plottingError = vn.get_plotly_figure(plotly_code=st.session_state["tempCode"] , df=st.session_state["df"])
        elif st.session_state['plottingLib'] =='Altair':
            st.session_state["fig"], plottingError = vn.get_altair_figure(altair_code=st.session_state["tempCode"] , df=st.session_state["df"])
        elif st.session_state['plottingLib'] =='Bokeh':
            st.session_state["fig"], plottingError = vn.get_bokeh_figure(bokeh_code=st.session_state["tempCode"] , df=st.session_state["df"])
        else:
            st.session_state["fig"], plottingError = vn.get_plotly_figure(plotly_code=st.session_state["tempCode"] , df=st.session_state["df"])

        st.session_state.messages.append({"role": "assistant", "fig": st.session_state["fig"] ,'code':st.session_state["tempCode"], 'figtype': st.session_state['plottingLib'], "type":"figure-code" })
        if plottingError is not None:
            st.session_state.messages.append({"role": "assistant", "content": f"An error occured when generating the figure above:\n {plottingError}" , 'type':'error' })
        st.rerun()
    elif st.session_state.get("show_chart", True):
        if st.session_state["fig"] is not None:
            if st.session_state['plottingLib'] == 'Plotly':
                st.session_state["fig"], plottingError = vn.get_plotly_figure(plotly_code=st.session_state["tempCode"] , df=st.session_state["df"])
            elif st.session_state['plottingLib'] =='Altair':
                st.session_state["fig"], plottingError = vn.get_altair_figure(altair_code=st.session_state["tempCode"] , df=st.session_state["df"])
            elif st.session_state['plottingLib'] =='Bokeh':
                st.session_state["fig"], plottingError = vn.get_bokeh_figure(bokeh_code=st.session_state["tempCode"] , df=st.session_state["df"])
            else:
                st.session_state["fig"], plottingError = vn.get_plotly_figure(plotly_code=st.session_state["tempCode"] , df=st.session_state["df"])

            st.session_state.messages.append({"role": "assistant", "content": f"For your question: {st.session_state['prompt']} - I was able to generate this figure:"  , "type":"markdown"})
            st.session_state.messages.append({"role": "assistant", "content": st.session_state["fig"]  , 'figtype':st.session_state['plottingLib'], 'type':'figure' })
        else:
            st.session_state.messages.append({"role": "assistant", "content": f"An error occurred when generating the figure above:\n {plottingError}" , 'type':'error' })
        st.rerun()
elif st.session_state["tempCode"]  is not None and st.session_state["code"]  is  None  and st.session_state["figureInstructions"] is None and st.session_state['enableUserTextInput'] == False:
    vn.logInfo('Confirming with user on next steps after generating Figure') 
    with st.chat_message("user"):
        plotyRadioInput = st.radio(
                "I would like to ...",
                options=["Instruct Changes (Figure):speaking_head_in_silhouette:","Edit Code Manually :pencil2:", "OK :white_check_mark:",'Instruct Changes (SQL) :rewind:',"Restart with a New Question :wastebasket:"],
                index=None,
                captions = ["Tell Data GENIE how the figure should be updated","Edit the Plot code", "Generated plot looks good - All Done",'Go back a step and tell Data GENIE to modify the SQL query',"Clear current question and start over"],
                horizontal = True
            )
    
    st.session_state['textInputHelp'] =plotyRadioInput
    if plotyRadioInput == "Edit Code Manually :pencil2:":
        
        st.warning("Please fix the generated Python code. Once you're done hit Ctrl + Enter to submit")
        python_code_response = code_editor(st.session_state["tempCode"], lang="python")
        fixed_python_code = python_code_response["text"]

        if fixed_python_code != "":
            st.session_state.messages.append({"role": "user", "content": "I would like to Edit the Plot Code :pencil2:", 'type':"markdown" })
            st.session_state["code"] = "#Edited Python Code:\n"+python_code_response["text"]
            st.session_state.messages.append({"role": "user", "content": st.session_state["code"], 'type':'code' })
            st.rerun()
        else:
            st.stop()
    elif plotyRadioInput == "OK :white_check_mark:":
        st.session_state.messages.append({"role": "user", "content": "Plot code looks good! :white_check_mark:", 'type':'markdown'   })
        st.session_state.messages.append({"role": "assistant", "content": "Do you want to save this question and SQL answer-pair to my knowledge base?"  , "type":"markdown"})
        st.session_state["code"] = st.session_state["tempCode"]
        st.rerun()
    elif plotyRadioInput == 'Instruct Changes (Figure):speaking_head_in_silhouette:':
        vn.logInfo('entering Instruct Changes (Figure)')   
        st.session_state['enableUserTextInput'] = True
        st.session_state["fig"] = None
        st.rerun()
    elif plotyRadioInput == 'Instruct Changes (SQL) :rewind:':
        vn.logInfo('entering Instruct Changes (SQL)')   
        st.session_state.messages.append({"role": "assistant", "content": "Here is the SQL query we have been using, let me know how you wish to change it:"  , "type":"markdown"})
        st.session_state.messages.append({"role": "assistant", "sql":  st.session_state['sql'] , "type":"sql"})
        st.session_state['enableUserTextInput'] = True
        st.session_state["tempCode"] =None
        st.session_state["fig"] = None
        st.session_state["df"] =None
        st.session_state["tempSQL"] =None
        st.session_state["sql"] =None
        st.session_state["code"]=None
        st.rerun()
    elif plotyRadioInput == "Restart with a New Question :wastebasket:":
        st.session_state.messages.append({"role": "user", "content": "Lets restart with a new question"  , "type":"markdown"})
        st.session_state.messages.append({"role": "assistant", "content": "Got it, Im ready for your next question. Please go ahead and ask a new question...", 'type':'markdown'   })
        resetPrompt()
    else:
        st.stop()
        
elif st.session_state["fig"] is not None and st.session_state["saveQnAPair"] is None:

    vn.logInfo('Confirm with user if the question and sql answer should be saved')   
    with st.chat_message("user"):
        plotyRadioInput = st.radio(
                "I would like to ...",
                options=["Yes :floppy_disk:", "No :x:"],
                index=None,
                captions = ["Yes, Save the question and SQL answer-pair", "No, do not save"],
                horizontal = True
            )
    st.session_state['textInputHelp'] =plotyRadioInput

    if plotyRadioInput == "Yes :floppy_disk:":
        st.session_state.messages.append({"role": "user", "content": "Yes, Save the question and SQL answer-pair :floppy_disk:", 'type':"markdown" })
        vn.add_question_sql(question = st.session_state["prompt"], sql=st.session_state["sql"],schema= st.session_state["vnModel"])
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