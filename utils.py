import json
import streamlit as st
import os
import re
from streamlit_lottie import st_lottie

def load_lottie_file(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)
    
def string_regex(string: str, regex: str):
    return re.search(regex, string)
    
def footer():
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #e6e6fa;
            color: #4f4f4f;
            text-align: center;
            padding: 10px;
            font-size: 20px;
            border-top: 1px solid #dcdcdc;
        }
        </style>
        <div class="footer">
            <p>
                Created by MTS | <a href="https://github.com/mtsfernando" target="_blank">GitHub</a> | Built with the Lankan Spirit
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )