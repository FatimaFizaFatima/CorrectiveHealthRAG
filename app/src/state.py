from langgraph.graph import MessagesState

class AgentState(MessagesState):
    retry_count : int
    filename : list[str]