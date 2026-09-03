from fastapi import APIRouter, Depends
from app.schema.corrective_health_schema import QuerySchema, ResponseSchema
from app.model.corrective_health_model import CorrectiveRAGModel
from app.src.graph import graph
from langchain_core.messages import HumanMessage
from app.db.session import get_db
from sqlalchemy.orm import Session


router = APIRouter(prefix='/corrective-health', tags=["CorrectHealthRAG"])

@router.post('/', response_model=ResponseSchema)
def query_rag(request: QuerySchema, db: Session = Depends(get_db)):

    state = {'messages': [HumanMessage(content=request.question)], 'retry_count': 0}
    result = graph.invoke(state)
    answer = result["messages"][-1].content
    filename = result.get("filename", [])

    log_entry = CorrectiveRAGModel(
        filename=", ".join(filename),
        question=request.question,
        answer=answer,
    )
    db.add(log_entry)
    db.commit()

    return ResponseSchema(answer=answer)