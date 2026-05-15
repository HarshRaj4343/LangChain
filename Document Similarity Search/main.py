from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
load_dotenv()   

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

document = [
    "Virat Kohli is an Indian cricketer.",
    "Sachin Tendulkar is a former Indian cricketer.",
    "Rohit Sharma is an Indian cricketer.",
    "MS Dhoni is a former Indian cricketer.",
    "Sourav Ganguly is a former Indian cricketer."
]

query = "Tell me about Sourav Ganguly."
# Embed the documents and the query
doc_embeddings = embeddings.embed_documents(document)
query_embedding = embeddings.embed_query(query)
# Compute cosine similarity between the query and each document
similarities = cosine_similarity([query_embedding], doc_embeddings)
print("Cosine Similarities:", similarities)
print("Most similar document:", document[np.argmax(similarities)])

