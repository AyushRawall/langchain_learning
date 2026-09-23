from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text_splitter=SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)
text="""Horticulture is the science and art of growing fruits, vegetables, flowers, and ornamental plants.
It plays an important role in agriculture, food production, and maintaining a healthy environment.MS Dhoni is a former Indian cricketer and one of India's most successful cricket captains.He is known for his calm leadership, excellent wicketkeeping, and finishing skills.

The Land Rover Defender is a famous luxury off-road SUV known for its rugged design and strong performance.
It offers excellent off-road capability and can handle challenging terrains with ease.
The Defender combines adventure-focused performance with modern technology and premium comfort.
It is available in multiple body styles and engine options to suit different driving needs."""

docs=text_splitter.create_documents([text])

print(len(docs))

print(docs)