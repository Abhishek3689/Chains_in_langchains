import os,time
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint,HuggingFacePipeline,ChatHuggingFace
from langchain.schema.runnable import RunnableParallel,RunnablePassthrough

start=time.time()
load_dotenv()

os.environ['HF_HOME'] = 'D:\Data_Science_Learnings\Langchain'

llm=HuggingFacePipeline.from_model_id(
    model_id='Qwen/Qwen2-0.5B-Instruct',
    task='text-generation',
    device=-1,  # Force CPU
    pipeline_kwargs={
        'max_new_tokens': 300,  # Increased to avoid truncation
        'temperature': 0.7,
        'do_sample': True }
)

prompt1=PromptTemplate(template="Give me a detailed description about {topic}",
                       input_variables=['topic'])

prompt2=PromptTemplate(template="Give me a summary of the following text \n{text}",
                       input_variables=['text'])

parser=StrOutputParser()

chain=prompt1 | llm | parser |{'text': RunnablePassthrough()}| prompt2 | llm | parser

res=chain.invoke({'topic':'tree'})
print(res)
chain.get_graph().print_ascii()

end=time.time()
print(end-start)