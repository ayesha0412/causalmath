import os
import time
from dotenv import load_dotenv
import httpx
from typing import List, Dict, Union
from volcenginesdkarkruntime import Ark


load_dotenv()
# Constants (adjust these based on your needs)
ARK_MAX_RETRY = 10000   # Maximum number of retry attempts
ARK_RETRY_SLEEP = 2     # Seconds to sleep between retries

def ark_api_caller(
    messages: List[Dict[str, str]],
    API_KEY: str = "",  # If provided, used for authentication; otherwise, rely on env variables.
    model: str = "ep-20250218143759-s65bj",
    stream: bool = True
) -> Union[str, tuple]:
    """
    Calls the Ark API with the provided messages and handles retries/errors.
    
    Returns:
        - If streaming is enabled, returns a tuple (reasoning_content, content) – both strings.
        - If non-streaming, returns a tuple with (reasoning_content, content),
        where reasoning_content may be an empty string if not provided.
        - In case of failure, returns "API_ERROR_OUTPUT".
    """
    
    output = "API_ERROR_OUTPUT"
    
    for _ in range(ARK_MAX_RETRY):
        try:
            # Initialize the Ark client; pass API_KEY if available.
            if API_KEY:
                client = Ark(api_key=API_KEY, timeout=httpx.Timeout(timeout=1800))
            else:
                client = Ark(timeout=httpx.Timeout(timeout=1800))
            
            # Make the API call with the desired stream setting.
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=stream
            )
            
            if stream:
                # For streaming responses, accumulate tokens as they arrive.
                final_reasoning = ""
                final_content = ""
                
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta:
                        # Extract tokens safely (default to empty strings if missing)
                        token_reasoning = getattr(delta, "reasoning_content", "") or ""
                        # token_content = getattr(delta, "content", "") or ""
                        
                        final_reasoning += token_reasoning
                        # final_content += token_content
                        
                        # Optionally, print the chunk as it arrives
                        # print(token_reasoning + token_content, end='', flush=True)
                        print(token_reasoning, end='', flush=True)
                
                # output = (final_reasoning, final_content)
                output = final_reasoning   # For non-streaming, return the final reasoning only.
                break  # Exit retry loop on success
                
            else:
                # Non-streaming: extract the complete response directly.
                if response and response.choices and len(response.choices) > 0:
                    message_obj = response.choices[0].message
                    reasoning_content = getattr(message_obj, "reasoning_content", "") or ""
                    # content = getattr(message_obj, "content", "") or ""
                    
                    # output = (reasoning_content, content)
                    output = reasoning_content  
                    break  # Exit retry loop on success

        except Exception as e:
            print("Ark API call error:", type(e), e)
            time.sleep(ARK_RETRY_SLEEP)
    
    print("\nArk API output:", output)
    return output


api_key = os.environ.get("ARK_API_KEY")
messages = [
        {"role": "system", "content": "你是人工智能助手"},
        {"role": "user", "content": "常见的十字花科植物有哪些？"},
    ]
model = "deepseek-r1-250120"
output = ark_api_caller(messages, API_KEY=api_key, model=model)
