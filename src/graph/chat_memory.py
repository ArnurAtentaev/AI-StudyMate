import os
import uuid
from dotenv import load_dotenv

from langchain_postgres import PostgresChatMessageHistory, PGEngine
from sqlalchemy import create_engine
import psycopg

load_dotenv(".env")
PG_USER = os.getenv("POSTGRES_USER")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD")
PG_DATABASE = os.getenv("POSTGRES_DB")
PG_PORT_EXPOSED = os.getenv("PG_PORT_EXPOSED")

connection = psycopg.connect(
    dbname=PG_DATABASE,
    user=PG_USER,
    password=PG_PASSWORD,
    host="localhost",
    port=PG_PORT_EXPOSED
)

table_name = "chat_history"

session_id = str(uuid.uuid4())
table_name = "chat_history"
PostgresChatMessageHistory.create_tables(connection, table_name)
