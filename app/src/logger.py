import logging
import functools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agentic_rag")

def log_node(func):
    @functools.wraps(func)
    def wrapper(state, *args, **kwargs):
        node_name = func.__name__
        logger.info(f"ENTER {node_name} | retry_count={state.get('retry_count', 0)}")

        result = func(state, *args, **kwargs)

        if isinstance(result, dict) and "messages" in result:
            last_msg = result["messages"][-1]
            preview = str(last_msg.content)[:100] or "(tool_call, no text content)"
            logger.info(f"EXIT  {node_name} | output: {preview}")
        else:
            logger.info(f"EXIT  {node_name} | routed to: {result}")

        return result
    return wrapper