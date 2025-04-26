import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain.schema.runnable import RunnableParallel,RunnablePassthrough
from pydantic import BaseModel,Field
from typing import Literal


load_dotenv()
os.environ['HF_HOME'] = 'D:\Data_Science_Learnings\Langchain'

class Feedback(BaseModel):
    sentiment:Literal['positive','negative']=Field(description="Give the sentiment of the feedback")

parser=PydanticOutputParser(pydantic_object=Feedback)


llm=HuggingFacePipeline.from_model_id(
  model_id='Qwen/Qwen2-0.5B-Instruct',
    task='text-generation',
    device=-1,  # Force CPU
    pipeline_kwargs={
        'max_new_tokens': 300,  # Increased to avoid truncation
        'temperature': 0.7,
        'do_sample': True }
)

prompt1=PromptTemplate(
    template="Classify the sentiment of the following feedback into positive or negative \n{feedback}\n{format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

classifier_chain=prompt1 | llm | parser
res=classifier_chain.invoke({'feedback':'This mobile is awesome'})

print(res)