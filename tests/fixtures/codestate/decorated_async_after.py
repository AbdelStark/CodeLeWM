import httpx


def trace(fn):
    return fn


@trace
async def fetch(client, url):
    response = await client.get(url, timeout=5)
    return response.json()
