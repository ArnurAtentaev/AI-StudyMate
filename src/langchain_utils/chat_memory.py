import os
import uuid
from dotenv import load_dotenv

from langchain_postgres import PostgresChatMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory
import psycopg2
from sqlalchemy.engine.url import URL

load_dotenv(".env")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD")
PG_USER = os.getenv("POSTGRES_USER")
PG_DATABASE = os.getenv("POSTGRES_DB")
PG_PORT_SERVICE = os.getenv("PG_PORT_SERVICE")

conn_info = "postgresql+psycopg2://postgres:12345678@localhost:5432/chat_history"
sync_connection = psycopg2.connect(conn_info)

# Create the table schema (only needs to be done once)
table_name = "chat_history"
PostgresChatMessageHistory.create_tables(sync_connection, table_name)
