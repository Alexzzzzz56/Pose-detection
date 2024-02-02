import streamlit as st
import os
# HKU SPACE CCIT4080
# Team: ∀ ASS
# FINISHED by <Li Ho Yin>


st.set_page_config(page_title="∀ ASS", page_icon="For_all_ASS.jpeg", layout="centered")
col1, col2 = st.columns([1, 8])
col1.image("For_all_ASS.jpeg")
col2.title("∀ ASS")
with st.sidebar:
    st.image("For_all_ASS.jpeg")
    st.title("∀ ASS Team members")
    st.header("", divider="red")
    mem1, mem2, mem3, mem4 = st.columns([1, 1, 1, 1])
    mem1.write("Angus Li")
    mem2.write("Alex Lau")
    mem3.write("Sunny Yau")
    mem4.write("Sunny Chan")
    st.header("", divider="red")
st.header("", divider=True)
st.subheader("Hello! We are team _∀ ASS_ and the students from HKU SPACE!")
st.image("Project_idea.jpg", caption="∀ ASS Project idea")
