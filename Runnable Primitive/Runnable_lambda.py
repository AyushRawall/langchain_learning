from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    task="text-generation"

)

model=ChatHuggingFace(llm=llm)

parser=StrOutputParser()

prompt=PromptTemplate(
    template="give a joke on {topic}",
    input_variables=["topic"]

)


joke_gen_chain=RunnableSequence(prompt,model,parser)

parallel_chain=RunnableParallel({
    "joke":RunnablePassthrough(),
    "word_count":RunnableLambda(lambda x:len(x.split()))
})

final_chain=RunnableSequence(joke_gen_chain,parallel_chain)


result=final_chain.invoke({"topic":"bikes"})

final_result="""{} \n word_count -{}""".format(result["joke"],result["word_count"])

print(final_result)