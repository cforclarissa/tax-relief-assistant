import streamlit as st
from PIL import Image
import os

st.logo(
    image="images/reliefguide_logo.png",
    size="large"
)

st.title("Methodology")

st.markdown("""
          
The ReliefGuide Tax Relief Assistant is built using Streamlit for the frontend UI, OpenAI for LLM and embeddings, and FAISS for vector storage. The system employs LangChain's RetrievalQA chain to facilitate retrieval-augmented generation (RAG) for accurate and contextually relevant answers.
            
### Core Technologies
            
Frontend UI: Streamlit
            
GUI Components: Streamlit file uploader, text input, expanders
                        
Authentication: Streamlit session-state role verification
            
LLM & Embeddings: OpenAI (gpt-4o-mini, text-embedding-3-small)
                        
Vector Store: FAISS
            
Retrieval: LangChain RetrievalQA chain
                        
Text Splitting: LangChain RecursiveCharacterTextSplitter
            
Guardrails: Regex-based prompt injection detection & output sanitization                   
     
""")

st.subheader("1. Admin Flow")
st.image(image="images/Admin_Flow.png", use_container_width=True)

st.markdown("""
### Details of Admin Flow

Step 1 — Authentication
            
Admin selects the Admin role and submits credentials to log in.
Once authenticated, admin can open the Upload Tax Relief Knowledge Base page.

Step 2 — CSV Upload
            
Admin uploads the tax-relief FAQ CSV, which includes:
- Category
- Question
- Official Answer

Step 3 — Data Processing

The system:
- Saves the file to a temporary directory
- Reads the file using encoding fallbacks
- Converts all content into plain text
- Splits text into small semantic chunks for embedding

Step 4 — Embedding & Indexing
            
The system:
- Uses text-embedding-3-small to embed chunks
- Saves vectors into a shared FAISS index (./faiss_db)
- This becomes the knowledge base for all user queries

Step 5 — Validation & Error Surfacing
            
Any ingestion errors are shown in the UI and logged to the console.            
  
""")

st.subheader("2. User Flow")
st.image(image="images/User_Flow.png", use_container_width=True)

st.markdown("""
          
### Details of User Flow
            
Step 1 — User Login
User logs in through the same Login page, selecting the “User” role. 
Once authenticated, user can open the Tax Relief Assistant page.

Step 2 — System Initialization
            
Before user interacts, the system:
- Loads prompt-injection guardrails
- Applies output sanitization policies
- Loads the FAISS vector index
- Prepares a RetrievalQA chain using gpt-4o-mini

Step 3 — Query Execution

When the user submits a question:
- Suspicious inputs are blocked
- Otherwise, the query → FAISS retrieval → LLM answer
- Response is sanitized (removal of unsafe or off-topic content)
- The final answer is displayed in the chat interface, with feedback options to thumbs up/down.
- Chat history is updated
- Only the last 3 chat exchanges are stored/displayed

Step 4 — Runtime Error Handling

Any failures raise a Streamlit UI error and log to console.            

""")

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

