from typing import Optional,TypedDict
from pydantic import BaseModel,Field,EmailStr

class emp(BaseModel):
    name:str
    email:EmailStr
    cgpa:float=Field(gt=0,lt=10,default=5,description="A Decimal value representing Cgpa of student ")
    age:Optional[int]=None

new_student={'name':'Abhishek','age':34,'email':'abhi@gmail.com'}

emp1=emp(**new_student)

print(emp1)
print(emp1.email)