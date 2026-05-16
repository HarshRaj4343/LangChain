from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational"
)
prompt1 = PromptTemplate(
    input_variables=['topic'],
    template = "Generate a detailed report on {topic}."
)

prompt2 = PromptTemplate(
    input_variables=['text'],
    template = "Summarize the following report: {text}"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({"topic": "Unemployment in India"})
print(result)