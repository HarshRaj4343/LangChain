# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import SystemMessage,AIMessage,HumanMessage

# chat_temp = ChatPromptTemplate([
#     SystemMessage(content = "You are a helpful {domain} expert."),
#     HumanMessage(content = "Explain in simple terms, what is {topic}")
# ])

# # this doesn't work since langchain is still in development stage.

# prompt = chat_temp.invoke({'domain':'cricket','topic':'Dusra'})
# print(prompt)


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage

chat_temp = ChatPromptTemplate.from_messages([  # from messages is optional
    ('system','You are a helpful {domain} expert.'),
    ('human','Explain in simple terms, what is {topic}')
])

# this doesn't work since langchain is still in development stage.

prompt = chat_temp.invoke({'domain':'cricket','topic':'Dusra'})
print(prompt)