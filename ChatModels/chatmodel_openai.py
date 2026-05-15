from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4",temperature=0.7,max_completion_tokens=1000) 
# temperature is a parameter that controls the randomness of the output. A higher temperature will result in more random output, while a lower temperature will result in more deterministic output. It lies between 0 and 2. 
result =model.invoke("What is the capital of France?")
print(result.content)

