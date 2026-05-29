from loader import yt_transcripter
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunker_transcript(docs,chunk_size,chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    return chunks

