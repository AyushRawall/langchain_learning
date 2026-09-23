from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser,ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="",
    task="text-generatQwen/Qwen2.5-Coder-7B-Instruction"
)

model=ChatHuggingFace(llm=llm)

schema=[
    ResponseSchema(name="benefit_1",description="benefit 1 about the topic"),
    ResponseSchema(name="benefit_2",description="benefit 2 about the topic"),
    ResponseSchema(name="benefit_3",description="benefit 3 about the topic"),
    ResponseSchema(name="benefit_4",description="benefit 4 about the topic")
]

parser=StructuredOutputParser.from_response_schemas(schema)

template=PromptTemplate(
    template="give 4 Benfits of the {topic} \n {format_instruction} ",
    input_variables=["topic"],
    partial_variables={"format_instruction":parser.get_format_instructions()}
)
# without chains
# prompt=template.invoke({"topic":"green tea"})

# result=model.invoke(prompt)

# final_result=parser.parse(result.content)

# print(final_result)

# with chains

chain=template | model | parser

result=chain.invoke({"topic":"Black coffee"})

print(result)