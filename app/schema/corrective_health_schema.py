from pydantic import BaseModel

class QuerySchema(BaseModel):
    question : str

class ResponseSchema(BaseModel):

    answer : str

