import os, uuid
import pandas as pd
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions
from datetime import datetime
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL, CHAT_MODEL, COLLECTION_NAME, VECTORSTORE_DIR, LOGS_DIR, LOG_FILE
from src.intent import extract_intent

os.makedirs(LOGS_DIR, exist_ok=True)

_client = None
_collection = None

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

def _get_collection():
    global _collection
    if _collection is None:
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY),
            model_name=EMBEDDING_MODEL
        )
        chroma_client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
        _collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=openai_ef)
    return _collection

SYSTEM_PROMPT = """You are Sofia, a friendly Toyota sales assistant at Toyota Canarias, serving the Canary Islands.
PERSONALITY: Warm, professional, enthusiastic about Toyota. You understand island life: short urban trips, warm climate, mountain roads, tourism and family lifestyle. Conversational, never robotic. Ask smart follow-up questions. Never pressure the customer.
WHEN RECOMMENDING: Base everything on the retrieved vehicle info given to you. Mention specific grades, features and specs from the dossiers. Connect features to customer needs. Hybrid is perfect for island commuting and low fuel cost. AWD is great for rural and mountain terrain.
On price: invite them to visit the showroom or book a test drive.
IMPORTANT: Only recommend vehicles you have retrieved context for. End every conversation by inviting them to Toyota Canarias."""

def retrieve_context(query: str, intent: dict, n: int = 4) -> str:
    collection = _get_collection()
    search_q = ' '.join(filter(None, [query, intent.get('fuel_preference',''), intent.get('body_type_preference',''), intent.get('use_case','')]))
    try:
        where = None
        hint = intent.get('model_mentioned')
        if hint: where = {'model': {'$contains': hint}}
        kwargs = dict(query_texts=[search_q], n_results=n)
        if where: kwargs['where'] = where
        results = collection.query(**kwargs)
    except Exception:
        results = collection.query(query_texts=[search_q], n_results=n)
    parts = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        parts.append(f"[{meta.get('model')} - {meta.get('section')}]\n{doc[:1500]}")
    return "\n\n---\n\n".join(parts)

def log_query(session_id, user_msg, response, intent):
    record = {
        'timestamp': datetime.now().isoformat(), 'session_id': session_id,
        'user_message': user_msg, 'assistant_response': response[:500],
        'intent': intent.get('intent',''), 'budget_mentioned': intent.get('budget_mentioned',''),
        'fuel_preference': intent.get('fuel_preference',''), 'body_type': intent.get('body_type_preference',''),
        'family_size': intent.get('family_size',''), 'use_case': intent.get('use_case',''),
        'features_mentioned': str(intent.get('features_mentioned',[])),
        'model_mentioned': intent.get('model_mentioned',''), 'sentiment': intent.get('sentiment',''),
    }
    df_new = pd.DataFrame([record])
    if os.path.exists(LOG_FILE):
        df = pd.concat([pd.read_csv(LOG_FILE), df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(LOG_FILE, index=False)

def get_reply(user_message: str, history: list, session_id: str) -> str:
    intent  = extract_intent(user_message)
    context = retrieve_context(user_message, intent)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": f"RETRIEVED VEHICLE INFO:\n{context}\n\n---\n\nCustomer says: {user_message}"})
    r = _get_client().chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.7, max_tokens=600)
    reply = r.choices[0].message.content
    log_query(session_id, user_message, reply, intent)
    return reply
