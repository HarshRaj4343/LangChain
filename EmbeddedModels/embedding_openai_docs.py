from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=16)

doc = [
    "Delhi is the capital of India.",
    "Paris is the capital of France.",]
result = embeddings.embed_documents(doc)
print([str(r) for r in result])