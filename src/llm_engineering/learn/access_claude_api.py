"""The following code is used to explore various functionalities of the Anthropic API and the reusable chat function.
It starts with initializing the Anthropic client and creating a sample message. 
It then demonstrates the use of system prompts, the reusable chat function, basic streaming, and handling JSON-only responses.
The JSON handling could be handled by using the stop sequence argument and the add_assistant_message function commented out 
in the helpers.py file as this API doesnt support the use of stop_sequence argument. Therefore, JSON handling needs to be done 
manually after receiving the response.
"""

from dotenv import load_dotenv
import os
from pathlib import Path
from anthropic import Anthropic
from llm_engineering.utils.helpers import add_user_message, chat, clean_json, get_anthropic_client, model
import json

#Initialize the Anthropic client with the API key
client = get_anthropic_client()


#Create a sample message using the Anthropic client
message = client.messages.create(
    model = model,
    max_tokens = 1000,
    messages = [
        {
            "role": "user",
            "content": "Explain CECL in one sentence"
        }
    ]
)

print(message.content[0].text)
#End of sample API client code


#Create a system prompt for the math tutor scenario
system_prompt = """
You are a patient math tutor.
Do not directly answer a student's questions.
Guide them to a solution step by step.
"""
message = client.messages.create(
    model = model,
    max_tokens = 1000,
    system = system_prompt,
    messages=[
        {
            "role": "user",
            "content": "How do I solve 2x + 5 = 11?"
        }
    ]
)
text_content = next((block.text for block in message.content if block.type == "text"), "No text response found")
print(text_content)
# End of math tutor scenario

#Using the reusable chat function for the math tutor scenario
system_prompt = """
You are a patient math tutor.
Do not directly answer a student's questions.
Guide them to a solution step by step.
"""
answer = chat(messages=[
    {
        "role": "user",
        "content": "How do I solve 2x + 5 = 11?"
    }
], model=model, system = system_prompt)

print(answer)

#End of reusable chat function scenario

#Basic streaming implementation

messages = []
add_user_message(messages, "Write a 1 sentence description of a fake database")

stream = client.messages.create(
    model=model,
    max_tokens=1000,
    messages=messages,
    stream=True
)

# for event in stream:
#     print(event)

#Rather than manually parsing events, you can use the SDK's simplified streaming interface that extracts just the text content
with client.messages.stream(
    model=model,
    max_tokens=1000,
    messages=messages
) as stream:
    # for text in stream.text_stream:
    #     print(text, end="")

    for text in stream.text_stream:
        # Send each chunk to your client
        pass
final_message = stream.get_final_message()
print(final_message.content[0].text)
#End of basic streaming implementation

#Handling JSON only response from the chat function

messages = []

add_user_message(messages, "Generate a very short event bridge rule as JSON only")

text = chat(messages = messages, model=model)

cleaned_text = clean_json(text)

#End of JSON only response handling

