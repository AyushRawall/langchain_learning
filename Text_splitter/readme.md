#  LangChain Text Splitters

This folder contains my practice and implementation of **Text Splitters in LangChain**.

Text Splitters are used to divide large documents into smaller **chunks** before sending them to embeddings, vector databases, or LLMs. Proper chunking helps improve retrieval quality and makes documents easier for LLMs to process.

---

#  Topics Covered

In this section, I explored the following text-splitting techniques:

* Length-Based Splitter
* Structure-Based Splitter
* Document-Structure-Based Splitter
* Semantic Chunker
* `chunk_size`
* `chunk_overlap`

---

# 1.  Length-Based Splitter

A length-based splitter divides text into chunks based on a specified **number of characters or tokens**.

In LangChain, `RecursiveCharacterTextSplitter` is commonly used for this purpose.

### Example

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print(chunks)
```

### Key Points

* Splits text according to length.
* `chunk_size` controls the maximum size of each chunk.
* `chunk_overlap` keeps some text from the previous chunk.
* Useful as a general-purpose text splitter.
* Commonly used in RAG applications.

---

# 2.  Structure-Based Splitter

A structure-based splitter divides text according to its **natural structure**, such as:

* Paragraphs
* Lines
* Sentences
* Characters
* Separators

`RecursiveCharacterTextSplitter` tries multiple separators in a hierarchy rather than blindly cutting text.

### Example

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_text(text)
```

### How it works

It tries to split using larger semantic boundaries first:

```text
Paragraph
   ↓
Line
   ↓
Word
   ↓
Character
```

This helps keep related text together whenever possible.

---

# 3.  Document-Structure-Based Splitter

Some documents have their own specific structure.

For example:

* Markdown → headings, paragraphs, lists
* HTML → headings and HTML elements
* Code → classes, functions, blocks
* JSON → keys and nested structures

LangChain provides specialized splitters for some of these formats.

### Example: Markdown

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3")
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

documents = splitter.split_text(markdown_text)
```

### Key Points

* Uses the structure of the document.
* Helps preserve relationships between sections.
* Useful when working with structured documents.
* Metadata can be created from headings or other structural elements.

---

# 4.  Semantic Chunker

A **Semantic Chunker** divides text based on the **meaning and semantic similarity** of sentences rather than simply using character count.

It uses embeddings to determine when the meaning of the text changes significantly.

### Basic idea

```text
Sentence 1 ── Sentence 2 ── Sentence 3
       Similar meaning
              ↓
          Same chunk

Sentence 4 ── Different topic
              ↓
          New chunk
```

### Example

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

splitter = SemanticChunker(embeddings)

documents = splitter.create_documents([text])
```

### Key Points

* Uses embeddings to understand semantic similarity.
* Creates chunks based on changes in meaning.
* Can produce more meaningful chunks than simple fixed-size splitting.
* Useful for semantic search and RAG systems.
* Requires an embedding model.

---

#  Important Parameters

## `chunk_size`

`chunk_size` determines the **maximum size of a text chunk**.

For example:

```python
chunk_size=500
```

means the splitter tries to create chunks of approximately **500 characters/tokens**, depending on the splitter and its length function.

### Simple idea

```text
Large Document
       ↓
┌─────────────┐
│   Chunk 1   │  ← 500
├─────────────┤
│   Chunk 2   │  ← 500
├─────────────┤
│   Chunk 3   │  ← 500
└─────────────┘
```

**Small chunk size:** More precise retrieval, but potentially less context.

**Large chunk size:** More context, but potentially less precise retrieval.

---

# `chunk_overlap`

`chunk_overlap` specifies how much content from one chunk should be repeated in the next chunk.

For example:

```python
chunk_size=500
chunk_overlap=50
```

means approximately **50 units of text** are shared between consecutive chunks.

```text
Chunk 1
────────────────────────────
        500
────────────────────────────

              ↓ 50 overlap

          Chunk 2
          ────────────────────────────
                  500
          ────────────────────────────
```

### Why use overlap?

Without overlap, important information at the boundary of two chunks could get separated.

```text
Chunk 1: "The model achieved an accuracy of..."

Chunk 2: "92% on the test dataset."
```

With overlap, some context is preserved between chunks.

### Simple rule

```text
Higher overlap → More context
Lower overlap  → Less redundancy
```

---

#  Comparison of Text Splitters

| Splitter                 | Main Idea                 | Useful For                  |
| ------------------------ | ------------------------- | --------------------------- |
| Length-Based             | Splits according to size  | General text                |
| Structure-Based          | Uses separators/structure | Paragraphs and natural text |
| Document-Structure-Based | Uses document format      | Markdown, HTML, code, etc.  |
| Semantic Chunker         | Splits based on meaning   | Semantic search & RAG       |

---

# Text Splitting in RAG

Text splitting is an important step in a typical RAG pipeline.

```text
Documents
    ↓
Document Loader
    ↓
Text Splitter
    ↓
Text Chunks
    ↓
Embeddings
    ↓
Vector Database
    ↓
Retriever
    ↓
LLM
    ↓
Answer
```

Good chunking can improve the quality of retrieved information and therefore improve the final response generated by the LLM.

---

#  Learning Outcome

Through this practice, I learned:

* Why text splitting is required in LangChain.
* How length-based splitting works.
* How recursive/structure-based splitting works.
* How document-specific structure can be used for chunking.
* How semantic chunking uses embeddings.
* The purpose of `chunk_size`.
* The purpose of `chunk_overlap`.
* The role of text splitters in RAG applications.

---

# 🚀 Next Steps

After Text Splitters, the next concepts I plan to explore are:

* Vector Stores
* Retrievers
* RAG

---

**Learning LangChain step by step **
