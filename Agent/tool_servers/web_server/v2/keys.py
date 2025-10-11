import os

serper_key_pool = [
    {
        "url": "https://google.serper.dev/search",
        "key": os.environ.get("WEB_SEARCH_SERPER_API_KEY")
    }
]

qwen_api_pool = [
    {
        "url": "https://api.uniapi.vip/v1",
        "key": os.environ.get("UNI_API_KEY"),
        "model": "qwen2.5-72b-instruct",
    },
    # other qwen api provider ...
]


jina_api_pool = [
    {
        "key": os.environ.get("JINA_API_KEY")
    }
]


def get_serper_api() -> dict:
    """
    Returns a random Serper API from the pool.
    """
    import random
    return random.choice(serper_key_pool)

def get_qwen_api() -> dict:
    """
    Returns a random Qwen API from the pool.
    """
    import random
    return random.choice(qwen_api_pool)


def get_jina_api() -> dict:
    """
    Returns a random Jina API from the pool.
    """
    import random
    return random.choice(jina_api_pool)