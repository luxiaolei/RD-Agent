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
from json_repair import repair_json
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langserve import RemoteRunnable


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
        Create a chat completion using RemoteRunnable.

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
            endpoint = f"{self.base_url}/proxyllm/pro"
        elif "flash" in model.lower():
            endpoint = f"{self.base_url}/proxyllm/flash"
        else:
            # default to pro
            endpoint = f"{self.base_url}/proxyllm/pro"

        # Convert messages to LangChain format
        langchain_messages = []
        for message in messages:
            if message["role"] == "system":
                langchain_messages.append(SystemMessage(content=message["content"]))
            elif message["role"] == "user":
                langchain_messages.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                langchain_messages.append(AIMessage(content=message["content"]))

        # Create RemoteRunnable
        remote_runnable = RemoteRunnable(endpoint, headers=self.headers)

        try:
            # Invoke RemoteRunnable
            result = remote_runnable.invoke({
                "messages": langchain_messages,
            })

            # Process the result
            content = result.content if hasattr(result, 'content') else str(result)
            
            # Handle JSON mode
            if kwargs.get("response_format", {}).get("type") == "json_object":
                try:
                    repaired_json = repair_json(content)
                    parsed_json = json.loads(repaired_json)  # type: ignore
                    
                    # Check if the parsed JSON is a list
                    if isinstance(parsed_json, list):
                        # Find the first dictionary in the list
                        content = next((item for item in parsed_json if isinstance(item, dict)), None)
                        if content is None:
                            raise ValueError("No dictionary found in the JSON list")
                    elif isinstance(parsed_json, dict):
                        content = parsed_json
                    else:
                        raise ValueError("Parsed JSON is neither a list nor a dictionary")
                    
                    content = json.dumps(content)  # Convert back to JSON string
                except Exception as e:
                    logger.warning(f"Failed to repair JSON: {e}")
                    content = json.dumps({"error": f"Invalid JSON response: {content}"})

            choices = [{
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"  # Assuming 'stop' as default finish reason
            }]
            
            # Assuming usage data is not available in this format
            usage = {}
            
            return ChatCompletionResponse(choices, usage)
        except Exception as e:
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
