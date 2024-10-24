import asyncio

import requests
from langserve import RemoteRunnable  # type: ignore

# Define the base URL for your server
BASE_URL = "http://185.150.189.132:8011"

# Define the authentication token
AUTH_TOKEN = "EAFramwork@2024"

# Create RemoteRunnable instances for your endpoints with authentication
proxyllm_pro = RemoteRunnable(f"{BASE_URL}/proxyllm/pro/", headers={"X-Token": AUTH_TOKEN})
proxyllm_flash = RemoteRunnable(f"{BASE_URL}/proxyllm/flash/", headers={"X-Token": AUTH_TOKEN})



async def main():
    # Test ProxyLLM Pro
    print("Testing ProxyLLM Pro:")
    result = proxyllm_pro.invoke("Tell me a joke about programming.")
    print(result)
    print("\n" + "-"*50 + "\n")

    # Test ProxyLLM Flash
    print("Testing ProxyLLM Flash:")
    result = proxyllm_flash.invoke("What's the difference between a cat and a comma?")
    print(result)
    print("\n" + "-"*50 + "\n")

    # Test async invocation
    print("Testing async invocation:")
    result = await proxyllm_pro.ainvoke("Explain quantum computing in simple terms.")
    print(result)
    print("\n" + "-"*50 + "\n")

    # Test streaming
    print("Testing streaming:")
    async for chunk in proxyllm_pro.astream("Tell me a short story about AI."):
        print(chunk, end="", flush=True)
    print("\n" + "-"*50 + "\n")

    # Test batch processing
    print("Testing batch processing:")
    inputs = ["Tell me about parrots", "Tell me about cats"]
    results = proxyllm_pro.batch(inputs)
    for result in results:
        print(result)
    print("\n" + "-"*50 + "\n")

    # Test embedding
    print("Testing embedding:")
    texts = ["Hello, world!", "LangChain is awesome!"]
    response = requests.post(f"{BASE_URL}/embeddings/embed", json=texts, headers={"X-Token": AUTH_TOKEN})
    print(response.json())
    print("\n" + "-"*50 + "\n")

    # Test query embedding
    print("Testing query embedding:")
    query = "What is the meaning of life?"
    response = requests.post(f"{BASE_URL}/embeddings/embed_query", json={"text": query}, headers={"X-Token": AUTH_TOKEN})
    print(response.json())
    print("\n" + "-"*50 + "\n")

    # Test embedding with task_type
    print("Testing embedding with task_type:")
    texts = ["Hello, world!", "LangChain is awesome!"]
    response = requests.post(f"{BASE_URL}/embeddings/embed", json={"texts": texts, "task_type": "retrieval_document"}, headers={"X-Token": AUTH_TOKEN})
    print(response.json())
    print("\n" + "-"*50 + "\n")

    # Test query embedding with task_type
    print("Testing query embedding with task_type:")
    query = "What is the meaning of life?"
    response = requests.post(f"{BASE_URL}/embeddings/embed_query", json={"text": query, "task_type": "retrieval_query"}, headers={"X-Token": AUTH_TOKEN})
    print(response.json())
    print("\n" + "-"*50 + "\n")

    # Test embedding without task_type (should use default)
    print("Testing embedding without task_type:")
    texts = ["This is a test", "Without specifying task_type"]
    response = requests.post(f"{BASE_URL}/embeddings/embed", json=texts, headers={"X-Token": AUTH_TOKEN})
    print(response.json())
    print("\n" + "-"*50 + "\n")

    # Test ProxyLLM Pro using requests
    print("Testing ProxyLLM Pro using requests:")
    response = requests.post(
        f"{BASE_URL}/proxyllm/pro/invoke",
        json={'input': 'Tell me a joke about programming.'},
        headers={"X-Token": AUTH_TOKEN}
    )
    print(response.json())
    print("\n" + "-"*50 + "\n")

    # Test ProxyLLM Flash using requests
    print("Testing ProxyLLM Flash using requests:")
    response = requests.post(
        f"{BASE_URL}/proxyllm/flash/invoke",
        json={'input': "What's the difference between a cat and a comma?"},
        headers={"X-Token": AUTH_TOKEN}
    )
    print(response.json()) # {'output': {'content': str, 'additional_kwargs': {}, 'response_metadata': {'prompt_feedback': {'block_reason': int, 'safety_ratings': list}, 'finish_reason': str, 'safety_ratings': list}, 'type': str, 'name': None, 'id': str, 'example': bool, 'tool_calls': list, 'invalid_tool_calls': list, 'usage_metadata': {'input_tokens': int, 'output_tokens': int, 'total_tokens': int}}, 'metadata': {'run_id': str, 'feedback_tokens': list}}
    print("\n" + "-"*50 + "\n")

    # Test batch processing using requests
    print("Testing batch processing using requests:")
    response = requests.post(
        f"{BASE_URL}/proxyllm/pro/batch",
        json={'inputs': ["Tell me about parrots", "Tell me about cats"]},
        headers={"X-Token": AUTH_TOKEN}
    )
    print(response.json())
    print("\n" + "-"*50 + "\n")

    # Test streaming using requests
    print("Testing streaming using requests:")
    with requests.post(
        f"{BASE_URL}/proxyllm/pro/stream",
        json={'input': "Tell me a short story about AI."},
        headers={"X-Token": AUTH_TOKEN},
        stream=True
    ) as response:
        for line in response.iter_lines():
            if line:
                print(line.decode('utf-8'), end="", flush=True)
    print("\n" + "-"*50 + "\n")





if __name__ == "__main__":
    asyncio.run(main())
