from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

parser=JsonOutputParser()

template=PromptTemplate(
    template="Give me five facts about{topic} /n {format_instruction}",
    input_variables=['topic'],
    partial_variables={"format_instruction":parser.get_format_instructions}
)
# # without chain
# prompt=template.invoke({'topic':' Royal Enfield Bullet'})

# result=model.invoke(prompt)

# final_result=parser.parse(result.content)

# print(final_result)

#With Chain 

chain=template | model | parser

result=chain.invoke({"topic":"Royal Enfield Bullet"})

print(result)
