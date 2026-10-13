from anthropic.types import ToolParam
from pathlib import Path
from llm_engineering.utils.helpers import chat, get_current_datetime, load_tool_schema, model, get_anthropic_client, add_duration_to_datetime, run_conversation

client = get_anthropic_client()
CURRENT_DIR = Path(__file__).parent.parent
get_current_datetime_schema = load_tool_schema(CURRENT_DIR / "schemas" / "tools" / "get_current_datetime_schema.yaml")
add_duration_to_datetime_schema = load_tool_schema(CURRENT_DIR / "schemas" / "tools" / "add_duration_to_datetime_schema.yaml")
# Make sure the YAML schema has all the required fields. ToolParam will raise an error if any required field is missing.
# Acts as a validation step to ensure the tool schema is complete and correct.
get_current_datetime_schema = ToolParam({
**get_current_datetime_schema,
})

add_duration_to_datetime_schema = ToolParam({
**add_duration_to_datetime_schema
})

# Send a message to the Anthropic API requesting the current time using the defined tool schema.
# The use of tool arise when model needs to access information from the future or real time data.
# For example, the current time is not known until the moment the request is made, so the model needs to use the tool to get this real-time information.
# A weather based function can be implemented similarly, where the model uses a tool to fetch real-time weather data using an API.
messages = []
messages.append({
    "role": "user",
    "content": "What is the exact time, formatted as HH:MM:SS?"
})

response = client.messages.create(
    model = model,
    max_tokens=1000,
    messages=messages,
    tools=[get_current_datetime_schema],
)

messages.append({
    "role": "assistant",
    "content": response.content
})

tool_block = next(b for b in response.content if b.type == "tool_use")

tool_output = get_current_datetime(**tool_block.input)

messages.append({
    "role": "user",
    "content": [{
        "type": "tool_result",
        "tool_use_id": tool_block.id,
        "content": tool_output,
        "is_error": False
    }]
})

final_response = client.messages.create(
    model=model,
    max_tokens=1000,
    messages=messages,
    tools=[get_current_datetime_schema]
)

print(final_response.content[0].text)

# Multi tool conversations
messages = [{"role": "user", "content": "Take current date and add 103 days to it"}]

messages = run_conversation(messages, tools=[
    get_current_datetime_schema,
    add_duration_to_datetime_schema,
])

final_blocks = messages[-1]["content"]
print("\n".join(b.text for b in final_blocks if b.type == "text"))