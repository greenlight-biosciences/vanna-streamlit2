import streamlit as st
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
