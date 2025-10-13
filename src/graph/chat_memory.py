import os
from dotenv import load_dotenv

from fastapi import Request, Response
from langchain_postgres import PostgresChatMessageHistory
import psycopg

load_dotenv(".env")
PG_USER = os.getenv("POSTGRES_USER")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD")
PG_DATABASE = os.getenv("POSTGRES_DB")
PG_PORT_EXPOSED = os.getenv("PG_PORT_EXPOSED")

TABLE_NAME = "chat_history"
CONNECTION = psycopg.connect(
    dbname=PG_DATABASE,
    user=PG_USER,
    password=PG_PASSWORD,
    host="localhost",
    port=PG_PORT_EXPOSED,
)


def db_connection(session_id):
    chat_history_db = PostgresChatMessageHistory(
        TABLE_NAME, session_id, sync_connection=CONNECTION
    )
    return chat_history_db
