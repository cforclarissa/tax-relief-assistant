import streamlit as st
import tempfile
import os
import pandas as pd
import chardet
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

st.logo(
    image="images/reliefguide_logo.png",
    size="large"
)


st.title("Upload Tax Relief Knowledge Base (Admin Only)")

# --- Login Check ---
if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    st.error("You must log in first.")
    st.stop()
elif st.session_state["role"] != "Admin":
    st.error("Access denied! Only Admin can view this page.")
    st.stop()

try:
    apikey = st.secrets["OPENAI"]["OPENAI_API_KEY"]

    # --- Helper function to read CSV with encoding detection ---
    def read_csv_with_fallback(path):
        try:
            return pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            with open(path, "rb") as f:
                raw = f.read()
                result = chardet.detect(raw)
                enc = result["encoding"] or "latin1"
            return pd.read_csv(path, encoding=enc)

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save the uploaded CSV file
            csv_path = os.path.join(tmpdir, "uploaded.csv")
            with open(csv_path, "wb") as f:
                f.write(uploaded_file.getvalue())


            st.success("CSV file uploaded!")

            # Collect CSV files
            csv_files = [csv_path]
            st.write(f"Found **{len(csv_files)}** CSV files.")

            documents = []
            for csv_file in csv_files:
                df = read_csv_with_fallback(csv_file)
                csv_text = df.to_csv(index=False)
                documents.append(csv_text)

            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = splitter.create_documents(documents)

            # --- Persist FAISS vectorstore ---
            persist_directory = "./faiss_db"  # permanent folder
            embeddings = OpenAIEmbeddings(api_key=apikey, model='text-embedding-3-small')
            vectorstore = FAISS.from_documents(
                documents=docs,
                embedding=embeddings
            )

            vectorstore.save_local(persist_directory)

            st.success("FAISS vectorstore created and persisted successfully!")
            st.write(f"FAISS database saved at: `{persist_directory}`")

except Exception as e:
    st.error("An error has occured, please inform the app creator.")
    print(f"Upload Tax Relief KB Page Error: {e}")

with st.expander("Disclaimer"):
    disclaimer = """
    **IMPORTANT NOTICE**  
    This web application is a prototype developed for educational purposes only.  
    The information provided here is **NOT** intended for real-world usage and should not be relied upon for making decisions, especially financial, legal, or healthcare matters.  

    Furthermore, please be aware that the LLM may generate inaccurate or incorrect information.  
    You assume full responsibility for how you use any generated output.  

    Always consult with qualified professionals for accurate and personalized advice. 
    """
    st.markdown(disclaimer)
