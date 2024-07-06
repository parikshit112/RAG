import openai
from dotenv import load_dotenv
import os
import streamlit as st
from utils import *

output = "Hi, How can I help you today?"
with st.sidebar:
    with st.form(key = 'querySubmission'):
        prompt = st.text_area(
            "QUESTION",
            placeholder="Enter question",
            key="1",
        )
        submit = st.form_submit_button()
        
        if prompt:
            output = QNA(prompt)
    
st.markdown(f'{output}</div>', unsafe_allow_html=True)