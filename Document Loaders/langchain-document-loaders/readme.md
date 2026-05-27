# 📝 Module 10: Document Loaders in LangChain

![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

## 🎯 Overview
This module covers the first major component of building Retrieval-Augmented Generation (RAG) applications: **Document Loaders**. Data exists in hundreds of formats (PDFs, URLs, CSVs, Databases). Document Loaders act as the ingestion layer, pulling data from these diverse sources and converting them into a standardized format that LangChain can process.

---

## 🧠 Core Architectural Concepts

Before writing any code, it is crucial to understand how LangChain handles data under the hood.

### 1. The `Document` Object
Regardless of the source (a webpage, a PDF, or a text file), every loader converts the data into a standardized LangChain `Document` object. This ensures downstream components (chunkers, embedders, retrievers) only have to deal with one data structure.

Every `Document` object contains two mandatory properties:
* **`page_content`**: The actual text extracted from the source.
* **`metadata`**: A dictionary containing contextual information (e.g., source file path, page number, author, creation date).

### 2. Memory Management: Eager vs. Lazy Loading
When dealing with massive datasets, how you load the data is just as important as what you load.
* **Eager Loading (`.load()`):** Loads the entire dataset into RAM at once and returns a Python `list`. Use this for single files or small datasets.
* **Lazy Loading (`.lazy_load()`):** Returns a Python `generator`. It loads one document into memory, processes it, and drops it before loading the next. **Essential for large-scale data pipelines** to prevent out-of-memory errors.

---

## 🛠️ Implementations & Code

*(Note: All loaders below are imported from the `langchain_community.document_loaders` package).*

