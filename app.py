import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

st.set_page_config(page_title="LangGraph Code Fixer", page_icon="🛠️", layout="centered")

st.title("🛠️ LangGraph Code Fixer")
st.write("Paste your broken Python code below, and Gemini will automatically debug and fix it.")

with st.sidebar:
    st.header("🔑 Configuration")
    api_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    st.markdown("[Get a Gemini API Key](https://aistudio.google.com/app/apikey)")

user_code = st.text_area("Enter your Python code here:", height=200, placeholder="def my_function():\n    print('Hello World'")

if st.button("🚀 Test & Auto-Fix Code"):
    if not api_key:
        st.error("Please provide a Gemini API Key in the sidebar!")
    elif not user_code.strip():
        st.warning("Please enter some code to fix!")
    else:
        st.subheader("💡 Suggested Fix & Explanation:")
        response_placeholder = st.empty()
        
        try:
            # Using gemini-2.0-flash for current API endpoints
            llm = ChatGoogleGenerativeAI(
                model="gemini-3.5-flash-lite",
                google_api_key=api_key,
                max_retries=2,
                timeout=15
            )
            
            prompt = f"Fix any errors in the following Python code and explain what was fixed:\n\n```python\n{user_code}\n```"
            
            full_response = ""
            for chunk in llm.stream(prompt):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
                
            response_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"Error calling Gemini API: {e}")