from langchain_text_splitters import RecursiveCharacterTextSplitter

text="""Horticulture is the science and art of growing fruits, vegetables, flowers, and ornamental plants.
It plays an important role in agriculture, food production, and maintaining a healthy environment.MS Dhoni is a former Indian cricketer and one of India's most successful cricket captains.He is known for his calm leadership, excellent wicketkeeping, and finishing skills.

The Land Rover Defender is a famous luxury off-road SUV known for its rugged design and strong performance.
It offers excellent off-road capability and can handle challenging terrains with ease.
The Defender combines adventure-focused performance with modern technology and premium comfort.
It is available in multiple body styles and engine options to suit different driving needs."""

splitter=RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0
)

chunks=splitter.split_text(text)

print(len(chunks))
print(chunks)