import streamlit as st
from app.rag import rag_fusion_generate_answer  # Make sure this path is correct

st.set_page_config(page_title="Shakespeare Chatbot", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
.chat-message {
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    max-width: 75%;
}
.user-message {
    background-color: #DCF8C6;
    align-self: flex-end;
    margin-left: auto;
}
.ai-message {
    background-color: #ECECEC;
    align-self: flex-start;
}
.stTextInput > div > div > input {
    border-radius: 20px;
    padding: 10px 15px;
}
</style>
""", unsafe_allow_html=True)

# Title & subtitle
st.title("🎭 Shakespeare Chatbot")
st.subheader("Ask Shakespeare anything about his works and life")

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Fixed emoji avatars
AVATARS = {
    "user": "🧑",
    "assistant": "🤖"
}

# Display past messages with fixed avatars
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=AVATARS.get(message["role"], "💬")):
        st.markdown(message["content"])

# New message input
if prompt := st.chat_input("Enter your question"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user input
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.spinner("Thinking..."):
            response = rag_fusion_generate_answer(prompt)
        st.markdown(response)
    
    # Store assistant reply
    st.session_state.messages.append({"role": "assistant", "content": response})