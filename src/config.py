import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL      = "gpt-4o"
COLLECTION_NAME = "toyota_canarias"
VECTORSTORE_DIR = "vectorstore"
PDF_DIR         = "data/dossiers"
LOGS_DIR        = "query_logs"
LOG_FILE        = os.path.join(LOGS_DIR, "query_log.csv")
