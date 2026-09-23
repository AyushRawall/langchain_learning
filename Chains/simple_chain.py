from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

prompt=PromptTemplate(
    template="Tell something about {topic}",
    input_variables=["topic"]
)

parser=StrOutputParser()

chain=prompt|model|parser

result=chain.invoke({"topic":"ms dhoni"})

print(result)