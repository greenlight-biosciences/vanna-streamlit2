import os
import streamlit as st
from dotenv import load_dotenv


@st.cache_resource(ttl=3600)
def setup_connexion():
    return 


def setup_session_state():
    st.session_state["my_question"] = None
