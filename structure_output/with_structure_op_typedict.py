from typing import TypedDict,Optional,Annotated,Literal
from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os,time

start=time.time()
load_dotenv()

os.environ['HF_HOME'] = 'D:\Data_Science_Learnings\Langchain'

## Use Local Model
# llm=HuggingFacePipeline.from_model_id(
#   model_id='Qwen/Qwen2-0.5B-Instruct',
#     task='text-generation',
#     device=-1,  # Force CPU
#     pipeline_kwargs={
#         'max_new_tokens': 300,  # Increased to avoid truncation
#         'temperature': 0.7,
#         'do_sample': True }
# )

# use Hugging face endpoint 
llm=HuggingFaceEndpoint(
    repo_id='mistralai/Mistral-7B-Instruct-v0.3',
    task='text-generation'
)
model=ChatHuggingFace(llm=llm)
# prompt=PromptTemplate(template="Write a small story about a {animal}",
#                       input_variables=['animal'])
# parser=StrOutputParser()

class Review(TypedDict):
    key_point:Annotated[list[str],"Write down all key points discussd in Reviews and return in form of list"]
    summary:Annotated[str,'give brief summary of reviews']
    sentiment:Annotated[Literal['positive','negative'],"Give the sentiment attached to the reviews"]
    pros:Annotated[Optional[list[str]],'provide all pros present in review in the form of list']
    cons:Annotated[Optional[list[str]],"Provide all cons in Reviews in form of list"]

structured_model=llm.with_structured_output(Review)

# class Review(TypedDict):
#     summary:str
#     sentiment:str

structure_output=model.with_structured_output(Review)


text="""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Abhishek Nishad"""

res=structure_output.invoke(text)
# chain=prompt | llm | parser
# res=chain.invoke({'animal':'cow'})
# # result=llm.invoke("Write a short poem about mango")
# # response=parser.parse(result)

print(res)
end=time.time()
print("Time Taken:",end-start)