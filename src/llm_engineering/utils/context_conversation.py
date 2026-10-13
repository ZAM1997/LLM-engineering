"""This script demonstrates a simple context-aware conversation using the reusable chat function.
It starts with an initial user question, gets a response, and then adds a follow-up question while maintaining the conversation context.
"""
from llm_engineering.utils.helpers import add_user_message, chat, add_assistant_message


messages = []

# Add the initial user question
add_user_message(messages, "Define quantum computing in one sentence")

# Get Claude's response
answer = chat(messages)

# Add Claude's response to the conversation history
# add_assistant_message(messages, answer)

# Add a follow-up question
add_user_message(messages, "Write another sentence")

# Get the follow-up response with full context
final_answer = chat(messages)