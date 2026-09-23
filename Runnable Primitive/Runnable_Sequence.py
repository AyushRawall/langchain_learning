from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    task="text-generation"

)

model=ChatHuggingFace(llm=llm)

parser=StrOutputParser()

prompt1=PromptTemplate(
    template="give a joke on {topic}",
    input_variables=["topic"]

)

prompt2=PromptTemplate(
    template="explain the following joke \n {text}",
    input_variables=["text"]
)

chain=RunnableSequence(prompt1,model,parser,prompt2,model,parser)

result=chain.invoke({"topic":"football"})

print(result)