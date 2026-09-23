from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough

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

joke_gen_chain=RunnableSequence(prompt1,model,parser)

parallel_chain=RunnableParallel({
    "joke":RunnablePassthrough(),
    "explanation":RunnableSequence(prompt2,model,parser)
})

final_chain=RunnableSequence(joke_gen_chain,parallel_chain)

result=final_chain.invoke({"topic":"bikes"})

print(result)