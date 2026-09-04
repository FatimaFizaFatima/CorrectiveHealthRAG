from pydantic import BaseModel, field_validator


class QuerySchema(BaseModel):
    question : str
    @field_validator('question')
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError('Question cannot be empty or whitespace only')
        return stripped

class ResponseSchema(BaseModel):

    answer : str

