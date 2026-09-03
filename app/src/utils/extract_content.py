
def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def get_latest_tool_messages(messages):
    latest = []
    for m in reversed(messages):
        if m.type == "tool":
            latest.insert(0, m)
        else:
            break
    return latest