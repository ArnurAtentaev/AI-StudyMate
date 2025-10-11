import os
from typing import Optional, Literal
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

CONFIG = load_dotenv(".env")
MODEL_LLM = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def initialize_llm(
    temperature: float | int,
):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        verbose=True,
        api_key=GROQ_API_KEY,
        temperature=temperature,
    )
    return llm
