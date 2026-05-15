# STATIC PROMPT UI

# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv
# import streamlit as st

# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-3.1-8B-Instruct",
#     task="conversational"
# )

# model = ChatHuggingFace(llm=llm)


# st.header("Research Assistant")
# user_input = st.text_input("Enter your research question:")
# if st.button('Summarise'):
#     result = model.invoke(user_input)
#     st.write(result.content)


# DYNAMIC PROMPT UI


# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from dotenv import load_dotenv
# import streamlit as st

# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-3.1-8B-Instruct",
#     task="conversational"
# )

# model = ChatHuggingFace(llm=llm)

# st.header("Research Assistant")
# paper_input = st.selectbox("Select a research paper:", ["Attention is all you need", "BERT", "GPT-3"])
# style_input = st.selectbox("Select a style:", ["Summary", "Bullet Points", "Key Takeaways"])
# length_input = st.selectbox("Select the length of the output:", ["Short", "Medium", "Lengthy"])
# template = PromptTemplate(
#     input_variables=["paper", "style", "length"],
#     template="""
# You are an expert research assistant.

# Your task is to summarize the research paper below.

# Research Paper:
# {paper}

# Instructions:
# - Use the following style: {style}
# - Keep the response length: {length}
# - Explain complex concepts simply
# - Highlight the most important contributions
# - Maintain clarity and structure

# Generate the response now.
# """

# )
# if st.button('Summarise'):
#     prompt = template.format(paper=paper_input, style=style_input, length=length_input,name="Research Assistant")
#     result = model.invoke(prompt)
#     st.write(result.content)


# ANOTHER DYNAMIC PROMPT UI


from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate,load_prompt
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational"
)

model = ChatHuggingFace(llm=llm)

st.header("Research Assistant")
paper_input = st.selectbox("Select a research paper:", ["Attention is all you need", "BERT", "GPT-3"])
style_input = st.selectbox("Select a style:", ["Summary", "Bullet Points", "Key Takeaways"])
length_input = st.selectbox("Select the length of the output:", ["Short", "Medium", "Lengthy"])

template = load_prompt("/Users/harshraj/Desktop/Langchain Models/template.json")

if st.button('Summarise'):
    chain = template | model
    result = chain.invoke({
        "paper": paper_input,
        "style": style_input,
        "length": length_input
    })
    st.write(result.content)



# Validation Of Input Variables in Prompt Template
