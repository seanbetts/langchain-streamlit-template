"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

from langchain import OpenAI, ConversationChain, PromptTemplate
from langchain.chains.conversation.memory import ConversationEntityMemory
from pydantic import BaseModel
from typing import List, Dict, Any


def load_chain():
    """Logic for loading the chain you want to use should go here."""
    template = """

        You are an enthusiastic conversational AI bot that remembers previous conversations.

        Given the following information, answer the question from the user using only that information. If you are unsure and the answer is not available, reply with "Sorry, I don't know how to help with that.”

        Context:
        {entities}

        Current conversation: {history}

        Human: {input}

        AI:

    """
    
    prompt = PromptTemplate(
    input_variables=["entities", "history", "input"],
    template=template,
    )
    
    llm = OpenAI(
    temperature=0
    )
    
    chain = ConversationChain(llm=llm, 
    verbose=False,
    prompt=prompt,
    memory=ConversationEntityMemory(llm=llm)
    )
    
    return chain

chain = load_chain()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = chain.run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
