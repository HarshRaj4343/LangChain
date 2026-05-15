# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
# from langchain_core.prompts import PromptTemplate,load_prompt
# from dotenv import load_dotenv
# import streamlit as st

# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-3.1-8B-Instruct",
#     task="conversational"
# )

# model = ChatHuggingFace(llm=llm)
# chat_history = []
# while True:
#     user_input = input("You: ")
#     chat_history.append(f"user: {user_input}")
#     if user_input.lower() in ["exit", "quit"]:
#         print("Exiting the chat. Goodbye!")
#         break
#     result = model.invoke(chat_history)
#     chat_history.append(f"assistant: {result.content}")
#     print(f"Assistant: {result.content}")

# print("Chat history:")
# for message in chat_history:
#     print(message)

# v2

# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
# from langchain_core.prompts import PromptTemplate,load_prompt
# from dotenv import load_dotenv
# import streamlit as st

# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-3.1-8B-Instruct",
#     task="conversational"
# )

# model = ChatHuggingFace(llm=llm)
# chat_history = [SystemMessage(content="You are a helpful assistant.")]
# while True:
#     user_input = input("You: ")
#     chat_history.append(HumanMessage(content=user_input))
#     if user_input.lower() in ["exit", "quit"]:
#         print("Exiting the chat. Goodbye!")
#         break
#     result = model.invoke(chat_history)
#     chat_history.append(AIMessage(content=result.content))
#     print(f"Assistant: {result.content}")

# print("Chat history:")
# for message in chat_history:
#     print(message)

# v3



from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_core.prompts import PromptTemplate,load_prompt
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational"
)

model = ChatHuggingFace(llm=llm)
chat_history = [SystemMessage(content="You are a helpful assistant.")]
while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chat. Goodbye!")
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print(f"Assistant: {result.content}")

print("Chat history:")
for message in chat_history:
    print(message)