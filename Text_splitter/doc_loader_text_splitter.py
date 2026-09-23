from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

loader=PyPDFLoader("Drones and Robotics.pdf")

docs=loader.load()

splitter=CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
    separator=" "
)

result=splitter.split_documents(docs)

print(result)