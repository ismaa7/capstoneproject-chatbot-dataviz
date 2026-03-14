import re, json
from openai import OpenAI
from src.config import CHAT_MODEL

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

def extract_intent(user_message: str) -> dict:
    prompt = (
        'Extract structured info from this car buyer query. '
        'Return ONLY valid JSON, no markdown, no extra text. '
        f'Query: "{user_message}" '
        'JSON fields: intent (browse/recommend/compare/spec_query/price_query/test_drive/general), '
        'budget_mentioned (null or number), fuel_preference (null or string), '
        'body_type_preference (null or string), family_size (null or number), '
        'use_case (null or string), features_mentioned (list), '
        'model_mentioned (null or string), sentiment (positive/neutral/negative).'
    )
    try:
        r = _get_client().chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300
        )
        raw = re.sub(r'```json|```', '', r.choices[0].message.content).strip()
        return json.loads(raw)
    except Exception as e:
        return {"intent": "general", "error": str(e)}
