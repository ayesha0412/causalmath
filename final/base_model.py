import pprint
import time
from typing import Dict, List
import openai



def gpt_api_caller(
    messages: List[Dict[str, str]],
    model: str = "gpt-35-turbo",
    temperature: float = 1.0
) -> str:

    client = openai.AzureOpenAI(
        azure_endpoint="https://feng-cloud-openai.openai.azure.com/",
        api_key="e51a662bf2934ff585b9e53b21b7f6c2",
        api_version="2024-02-15-preview"
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    generated_text = response.choices[0].message.content.strip()
    return generated_text



def llm_predict_gpt35(chain): 
    chain_text = " ".join(chain)
    user_prompt = (chain_text)
    
    client = openai.AzureOpenAI(
        azure_endpoint="https://feng-cloud-openai.openai.azure.com/",
        api_key="e51a662bf2934ff585b9e53b21b7f6c2",
        api_version="2024-02-15-preview"
    )

    response = client.chat.completions.create(
                model="gpt-35-turbo",
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_prompt}],
                temperature=1
            )
    
    generated_text = response.choices[0].message.content.strip()
    steps = generated_text.split('\n')
    return steps



if __name__ == "__main__":
    context = "2+4-5+7=?\n ### Step1: 6-5+7=?\n"
    result = llm_predict_gpt35(context)
    pprint.pprint(result)



