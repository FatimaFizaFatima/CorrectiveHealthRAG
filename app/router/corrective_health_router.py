from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.model.corrective_health_model import CorrectiveRAGModel
from app.schema.corrective_health_schema import (
    ApprovalSchema,
    QuerySchema,
    ResponseSchema,
)

router = APIRouter(prefix='/corrective-health', tags=["CorrectHealthRAG"])


def build_response(result: dict, thread_id: str, db: Session) -> ResponseSchema:
    interrupts = result.get("__interrupt__")
    if interrupts:
        actions = interrupts[0].value["actions"]
        return ResponseSchema(
            answer="This action needs human approval before it runs.",
            thread_id=thread_id,
            requires_approval=True,
            pending_actions=actions,
        )

    messages = result["messages"]
    answer = messages[-1].content
    filename = result.get("filename", [])
    tools_used = [
        call["name"]
        for m in messages
        if m.type == "ai" and m.tool_calls
        for call in m.tool_calls
    ]

    db.add(
        CorrectiveRAGModel(
            filename=", ".join(filename),
            question=messages[0].content,
            answer=answer,
        )
    )
    db.commit()

    return ResponseSchema(answer=answer, tools_used=tools_used, thread_id=thread_id)


@router.post('/', response_model=ResponseSchema)
async def query_rag(
    request: QuerySchema,
    http_request: Request,
    db: Session = Depends(get_db),
):
    graph = http_request.app.state.graph
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    state = {'messages': [HumanMessage(content=request.question)], 'retry_count': 0}
    result = await graph.ainvoke(state, config)

    return build_response(result, thread_id, db)


@router.post('/approve', response_model=ResponseSchema)
async def approve_action(
    body: ApprovalSchema,
    http_request: Request,
    db: Session = Depends(get_db),
):
    graph = http_request.app.state.graph
    config = {"configurable": {"thread_id": body.thread_id}}

    snapshot = await graph.aget_state(config)
    waiting = any(task.interrupts for task in snapshot.tasks)
    if not waiting:
        raise HTTPException(
            status_code=404,
            detail="No pending approval found for this thread_id.",
        )

    result = await graph.ainvoke(Command(resume={"approved": body.approved}), config)

    return build_response(result, body.thread_id, db)