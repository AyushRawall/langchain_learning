# LangChain Retrievers

This section contains my practice and learning work on **Retrievers in LangChain**.

Retrievers are used to retrieve relevant documents or information based on a user's query. They are an important component of **Retrieval-Augmented Generation (RAG)** systems because they help provide relevant context to an LLM.

##  Retrievers Covered

### 1. Wikipedia Retriever

The **Wikipedia Retriever** retrieves relevant information from Wikipedia based on a user's query.

It is useful when we want to retrieve general knowledge and factual information from Wikipedia.

---

### 2. MMR Retriever

**MMR (Maximum Marginal Relevance)** retrieves documents that are both relevant to the query and sufficiently different from each other.

It helps reduce duplicate or highly similar results and provides more diverse information.

**Main idea:**

```text
Query
  ↓
Find relevant documents
  ↓
Balance relevance + diversity
  ↓
Return diverse results
```

---

### 3. Multi-Query Retriever

The **Multi-Query Retriever** generates multiple variations of the user's original query and uses them to retrieve relevant documents.

This can improve retrieval when the original query may have multiple interpretations or when different wording can retrieve different relevant information.

**Workflow:**

```text
Original Query
      ↓
Generate multiple queries
      ↓
Retrieve documents
      ↓
Combine results
      ↓
Relevant documents
```

---

### 4. Vector Store Retriever

The **Vector Store Retriever** retrieves documents from a vector store based on the semantic similarity between the user's query and stored document embeddings.

The general workflow is:

```text
Documents
    ↓
Create Embeddings
    ↓
Store in Vector Database
    ↓
User Query
    ↓
Query Embedding
    ↓
Similarity Search
    ↓
Relevant Documents
```

This type of retriever is commonly used in **RAG applications**.

---

### 5. Contextual Compression Retriever

The **Contextual Compression Retriever** retrieves documents and then compresses the retrieved content so that only the information relevant to the query is passed forward.

This helps reduce unnecessary information in the retrieved context.

**Workflow:**

```text
User Query
    ↓
Retrieve Documents
    ↓
Compress Retrieved Content
    ↓
Keep Relevant Information
    ↓
Pass Context to LLM
```

It can be useful when retrieved documents contain a lot of information that is not directly related to the user's question.

---

##  Comparison

| Retriever                        | Main Purpose                                   |
| -------------------------------- | ---------------------------------------------- |
| Wikipedia Retriever              | Retrieve information from Wikipedia            |
| MMR Retriever                    | Retrieve relevant and diverse documents        |
| Multi-Query Retriever            | Generate multiple queries to improve retrieval |
| Vector Store Retriever           | Retrieve documents using vector similarity     |
| Contextual Compression Retriever | Retrieve and compress relevant information     |

##  What I Learned

Through these implementations, I learned:

* What retrievers are and why they are used.
* How LangChain can retrieve information from different sources.
* How semantic similarity is used for document retrieval.
* How MMR improves the diversity of retrieved documents.
* How multiple queries can improve retrieval results.
* How vector stores can be used for semantic search.
* How contextual compression can reduce irrelevant information.
* How retrievers are used as an important component of **RAG pipelines**.

##  Technologies Used

* Python
* LangChain
* LangChain Core
* Vector Stores
* Embeddings
* Wikipedia
* LLMs
* Retrieval-Augmented Generation (RAG)

##  Project Structure

```text
Retrievers/
│
├── wikipedia_retriever.py
├── MMR.py
├── multi_query_retriever.py
├── vector_store_retriever.py
├── contextual_compression_retriever.py
└── README.md
```


##  Purpose

This section is part of my **LangChain learning journey**, where I am implementing different concepts practically to understand how they work and how they can be used to build **RAG and LLM applications**.

More LangChain concepts and projects will be added as I continue learning.
