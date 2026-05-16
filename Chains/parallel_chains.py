from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import PromptTemplate
load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational"
)

llm2 = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="conversational"
)

model1 = ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    input_variables=['topic'],
    template = "Generate short and simple notes on \n {topic}."
)
prompt2 = PromptTemplate(
    input_variables=['topic'],
    template = "Create a 5 question quiz based on the following notes: {topic}"
)
prompt3 = PromptTemplate(
    template = "Merge the following notes and quiz into a single document: {notes} {quiz}",
    input_variables=['notes', 'quiz']
)
parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
}) 
merge_chain = prompt3 | model1 | parser
final_chain = parallel_chain | merge_chain

result = final_chain.invoke({"topic": "The French Revolution (1789–1799) was a radical political upheaval sparked by extreme economic hardship, high taxes, and deep social inequality under France's feudal system. Frustrated by a wealthy aristocracy and clergy who held all the power, the impoverished commoners overthrew King Louis XVI and established a republic based on the ideals of liberty, equality, and fraternity. This turbulent decade saw intense bloodshed, including the infamous Reign of Terror where thousands were executed by guillotine, and ultimately shattered the traditional authority of the monarchy and the Catholic Church. The revolution permanently altered global history by spreading democratic principles across Europe, finally concluding when military leader Napoleon Bonaparte seized control of the government."})
print(result)