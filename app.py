import streamlit as st
import pandas as pd
from datasets_search import search_datasets
from utils import load_dataset

st.title("AI Agent Driven AutoML Demo")

query = st.text_input("🔍 Search for a dataset", placeholder="e.g. iris, titanic, mnist")
if query:
    results = search_datasets(query)
    if results:
        choice = st.selectbox("Select a dataset:", results)
        if st.button("Load Dataset"):
            df = load_dataset(choice)
            st.write(df.head())
    else:
        st.warning("No datasets found!")
