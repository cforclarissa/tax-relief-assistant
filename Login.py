import base64
from pathlib import Path

import streamlit as st
import helper_functions.security as security

st.logo(
    image="images/reliefguide_logo.png",
    size="large"
)

st.title("Login")

st.sidebar.success("Login to access the chatbot. \n\nTo logout, click on Login page again")
message_placeholder = st.empty()  

def login():
    try:    
        # Choose role
        role = st.selectbox("Select Role", ["Admin", "User"])
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):            
            #print(f"Role:",role)
            if(security.verify_password(role, username, password)):
                st.session_state["logged_in"] = True
                st.session_state["role"] = role
                message_placeholder.success(role+" login successful! Use the sidebar to navigate.")
            else:                               
                raise ValueError("Invalid username or password")
                
    except Exception as e:
            message_placeholder.error("Invalid credentials")
            print(f"Login Page Error, Login: ",{e})

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["role"] = None

if not st.session_state["logged_in"]:
    login()
else:
    st.subheader(f"Welcome {st.session_state['role']}")
    st.write("You are logged in. Use the sidebar to navigate to your pages.")

    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.session_state["role"] = None
        st.rerun()



