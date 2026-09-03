from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_llm(temperature : float = 0):

    return ChatGoogleGenerativeAI(model = 'gemini-3.1-flash-lite', temperature=0)


