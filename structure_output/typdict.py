from typing import TypedDict,List,Optional

class Employee(TypedDict):
    name:str
    age:int
    dept:Optional[str]

new_employee:Employee={'name':'Abhishek','age':34,'dept':'DataScience'}
sec_emp:Employee={'name':'Nitesh','dept':'DataScience'}
print(sec_emp)