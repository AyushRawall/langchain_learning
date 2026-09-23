from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableBranch

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    task="text-generation"

)

model=ChatHuggingFace(llm=llm)

parser=StrOutputParser()

prompt1=PromptTemplate(
    template="give a Detailed Report  on {topic}",
    input_variables=["topic"]

)

prompt2=PromptTemplate(
    template="Generate a summary on following text \n {text}",
    input_variables=["text"]
)

Repot_gen_chain=RunnableSequence(prompt1,model,parser)

conditional_chain=RunnableBranch(
    (lambda x: len(x.split())>600,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

final_chain=RunnableSequence(Repot_gen_chain,conditional_chain)


result=final_chain.invoke({"topic":"bikes"})

print(result)