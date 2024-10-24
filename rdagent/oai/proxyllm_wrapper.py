"""
Write two classes, one is for embedding, one is for chat, that makes the interface same as
openai's client. but the inner implementation are using proxyllm remote server, we use requests to call the proxyllm server.

                response = self.embedding_client.embeddings.create(
                    model=self.embedding_model,
                    input=sliced_filtered_input_content_list,
                )
            for index, data in enumerate(response.data):
                content_to_embedding_dict[sliced_filtered_input_content_list[index]] = data.embedding



            kwargs = dict(
                model=self.chat_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=self.chat_stream,
                seed=self.chat_seed,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
            if json_mode:
                if add_json_in_prompt:
                    for message in messages[::-1]:
                        message["content"] = message["content"] + "\nPlease respond in json format."
                        if message["role"] == "system":
                            break
                kwargs["response_format"] = {"type": "json_object"}
            response = self.chat_client.chat.completions.create(**kwargs)


"""

"""ProxyLLM wrapper classes for embedding and chat that mimic the OpenAI client interface."""

import json
import os
from typing import Any, Dict, List, Optional, Union

import requests
from loguru import logger


class EmbeddingResponse:
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

class ChatCompletionResponse:
    def __init__(self, choices: List[Dict[str, Any]], usage: Dict[str, int]):
        self.choices = choices
        self.usage = usage

class EmbeddingsClient:
    def __init__(self, base_url: str, auth_token: str):
        """
        Initialize the EmbeddingsClient.

        Args:
            base_url (str): The base URL of the ProxyLLM server.
            auth_token (str): The authentication token for the ProxyLLM server.
        """
        self.base_url = base_url
        self.headers = {"X-Token": auth_token}

    def create(self, model: str, input: Union[str, List[str]], task_type: Optional[str] = None) -> EmbeddingResponse:
        """
        Create embeddings for the given input.

        Args:
            model (str): The model to use for embeddings (ignored, using ProxyLLM's default).
            input (Union[str, List[str]]): The input text(s) to embed.
            task_type (Optional[str]): The type of task for the embedding (e.g., "retrieval_document").

        Returns:
            EmbeddingResponse: The embedding response object.

        Raises:
            ValueError: If the input is not a string or a list of strings.
            requests.RequestException: If there's an error in the API request.
        """
        endpoint = f"{self.base_url}/embeddings/embed"
        
        if isinstance(input, str):
            payload = [input]
        elif isinstance(input, list) and all(isinstance(item, str) for item in input):
            payload = input
        else:
            raise ValueError("Input must be a string or a list of strings")

        if task_type:
            logger.warning("task_type parameter is not supported in the current API version")

        try:
            response = requests.post(endpoint, json=payload, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return EmbeddingResponse(data)
        except requests.RequestException as e:
            logger.error(f"Error creating embeddings: {e}")
            logger.error(f"Response content: {e.response.content if e.response else 'No response content'}")
            raise

class ChatCompletionClient:
    def __init__(self, base_url: str, auth_token: str):
        """
        Initialize the ChatCompletionClient.

        Args:
            base_url (str): The base URL of the ProxyLLM server.
            auth_token (str): The authentication token for the ProxyLLM server.
        """
        self.base_url = base_url
        self.headers = {"X-Token": auth_token}

    def create(self, model: str, messages: List[Dict[str, str]], **kwargs) -> ChatCompletionResponse:
        """
        Create a chat completion.

        Args:
            model (str): The model to use for chat completion. Use "pro" or "flash" in the name to select the endpoint.
            messages (List[Dict[str, str]]): The list of messages in the conversation.
            **kwargs: Additional parameters for the chat completion.

        Returns:
            ChatCompletionResponse: The chat completion response object.

        Raises:
            ValueError: If the model name doesn't contain "pro" or "flash".
        """
        if "pro" in model.lower():
            endpoint = f"{self.base_url}/proxyllm/pro/invoke"
        elif "flash" in model.lower():
            endpoint = f"{self.base_url}/proxyllm/flash/invoke"
        else:
            # default to pro
            endpoint = f"{self.base_url}/proxyllm/pro/invoke"

        payload = {
            "input": "\n".join([f"{m['role']}: {m['content']}" for m in messages]),
            **kwargs
        }

        # Add this block to handle JSON mode
        if kwargs.get("response_format", {}).get("type") == "json_object":
            payload["response_format"] = {"type": "json_object"}

        try:
            response = requests.post(endpoint, json=payload, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            # Ensure the response is in JSON format when requested
            if payload.get("response_format", {}).get("type") == "json_object":
                try:
                    json.loads(data["output"]["content"])
                except json.JSONDecodeError:
                    data["output"]["content"] = json.dumps({"error": "Invalid JSON response"})
            
            choices = [{
                "message": {
                    "role": "assistant",
                    "content": data["output"]["content"]
                },
                "finish_reason": data["output"].get("response_metadata", {}).get("finish_reason", '')
            }]
            usage = data["output"].get("usage_metadata", {})
            return ChatCompletionResponse(choices, usage)
        except requests.RequestException as e:
            logger.error(f"Error creating chat completion: {e}")
            raise

class ProxyLLMClient:
    def __init__(self, base_url: Optional[str] = None, auth_token: Optional[str] = None):
        """
        Initialize the ProxyLLMClient.

        Args:
            base_url (Optional[str]): The base URL of the ProxyLLM server. If not provided, it will be read from the PROXYLLM_BASE_URL environment variable.
            auth_token (Optional[str]): The authentication token for the ProxyLLM server. If not provided, it will be read from the PROXYLLM_AUTH_TOKEN environment variable.
        """
        self.base_url = base_url or os.getenv("PROXYLLM_BASE_URL", "http://185.150.189.132:8011")
        self.auth_token = auth_token or os.getenv("PROXYLLM_AUTH_TOKEN", "EAFramwork@2024")
        
        self.embeddings = EmbeddingsClient(self.base_url, self.auth_token)
        self.chat = ChatCompletionClient(self.base_url, self.auth_token)


if __name__ == "__main__":
    # Usage example:
    client = ProxyLLMClient()  # This will use environment variables or default values
    embedding_response = client.embeddings.create(model="text-embedding-ada-002", input="Hello, world!")
    chat_response = client.chat.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Tell me a joke"}])
    
    print(embedding_response)
    print(chat_response)
