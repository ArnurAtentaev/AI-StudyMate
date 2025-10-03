import os

from dotenv import load_dotenv

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings,
)
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.agents import (
    initialize_agent,
    AgentExecutor,
    AgentType,
    create_react_agent,
)


CONFIGS = load_dotenv(".env")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
MODEL_LLM = "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")

google_serper_search = GoogleSerperAPIWrapper(
    serper_api_key=SERPER_API_KEY, type="search", k=3
)

search_tool = Tool(
    name="Google Search",
    description="Searches Google for recent results.",
    func=google_serper_search.run,
)

llm = HuggingFaceEndpoint(
    repo_id=MODEL_LLM,
    task="conversation",
    temperature=0.5,
    top_k=10,
    top_p=0.9,
    huggingfacehub_api_token=HF_TOKEN,
)

chat = ChatHuggingFace(llm=llm, verbose=True)

system_prompt = """
Ты полезный ассистент. Который может подстраиваться под язык, на котором к тебе обращаются тем самым меняя язык на котором ты мыслишь и действуешь.
Также ты прикладываешь примеры, если они есть, не придумывай сам, а используй примеры которые ты находишь при поиске.
Если в найденном тексте нет примеров кода или использования, обязательно сделай дополнительный поиск с уточнением "example" или "пример"..
Важно: если в тексте встречаются технические термины (например: dict, class, agent, prompt, pipeline и т. д.), то ты НЕ переводишь их на русский, а оставляешь в оригинале.

Используй формат ReAct:
Thought: рассуждения
Action: название инструмента
Action Input: запрос
Observation: результат

Когда у тебя достаточно информации, ОБЯЗАТЕЛЬНО закончи ответ так:
Final Answer: [твой ответ]

Если ты сделал хотя бы один поиск, но уже можешь объяснить ответ — останавливайся!
"""

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        ("human", "{input}"),
        ("ai", "Инструменты, которые ты можешь использовать:\n{tools}"),
        ("ai", "Доступные названия инструментов: {tool_names}"),
        ("assistant", "{agent_scratchpad}"),
    ]
)

agent = create_react_agent(
    llm=chat,
    tools=[search_tool],
    prompt=prompt,
)
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True,
    max_iterations=3,
)

result = agent_executor.invoke(
    {"input": "Что такое dict в Python? Покажи на примерах."}
)
print(result)
