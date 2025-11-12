import streamlit as st
import requests
import time

# --- CONFIGURATION ---
# This is the public URL of your EKS Load Balancer.
# Make sure to replace this with your actual URL.
# The endpoint is /query, as we defined in api.py.
API_URL = "YOUR_URL_HERE/query"
# ---------------------

# Set up the Streamlit page
st.set_page_config(page_title="UTD Campus Assistant", page_icon="🎓")
st.title("🎓 UTD Campus Assistant")
st.caption("I have information about UTD's CS graduate tracks, courses, calendars, and professor ratings.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get new user input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare the request payload
    payload = {"question": prompt}

    # Display a "thinking" spinner and get the response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Send the request to your live EKS API
                response = requests.post(API_URL, json=payload, timeout=60)
                
                if response.status_code == 200:
                    # Parse the JSON response
                    data = response.json()
                    answer = data.get("answer", "Sorry, I received an unexpected response.")
                else:
                    # Show an error if the API call failed
                    answer = f"Error: Could not connect to the API. (Status code: {response.status_code})"
            except requests.exceptions.RequestException as e:
                # Show an error for network issues
                answer = f"Error: A network error occurred. {e}"
        
        # Display the assistant's answer
        st.markdown(answer)
    
    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})