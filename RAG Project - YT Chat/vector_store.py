from chunker import chunker_transcript
from langchain_huggingface import HuggingFaceEmbeddings
from loader import yt_transcripter
from langchain_chroma import Chroma
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

documents = yt_transcripter("https://youtu.be/MdeQMVBuGgY")


embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

chunks = chunker_transcript(documents,chunk_size=1000,chunk_overlap=200)

print("Number of chunks:", len(chunks))


vector_store = Chroma(
    embedding_function=embedder,
    persist_directory='project_db',
    collection_name='yt_chat'
)

# add documents
vector_store.add_documents(chunks)



base_retriever = vector_store.as_retriever(search_kwargs={"k": 5})


compressor = LLMChainExtractor.from_llm(model)


compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

query = "Based on Vijay's statements, what do you think about his overall mental picture of this scam?"
compressed_results = compression_retriever.invoke(query)


for i, doc in enumerate(compressed_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)


