from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "What is the capital of France?"
vector = embeddings.embed_query(text)
print(str(vector))

# Similarly, you can also embed documents and also you can also use HF's inference API to get embeddings from HuggingFace models.