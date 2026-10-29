import ast
from datetime import datetime, timedelta
import re
import json
import os
from anthropic import Anthropic
from dotenv import load_dotenv
from statistics import mean 
import yaml
from anthropic.types import Message
load_dotenv()

API_KEY = os.getenv("ANTHROPIC_API_KEY")
model = "claude-haiku-5-5"

def get_anthropic_client():
    API_KEY = os.getenv("ANTHROPIC_API_KEY")
    #Initialize the Anthropic client with the API key
    return Anthropic(api_key=API_KEY)


def add_user_message(messages, message):
    user_message = {"role": "user", 
                    "content": message.content if isinstance(message, Message) else message}
    messages.append(user_message)

def add_assistant_message(messages, text):
    assistant_message = {"role": "assistant", "content": text}
    messages.append(assistant_message)

def chat(messages, model, system=None, stop_sequences=None, tools=None,
         max_tokens=1000, return_message=False):
    params = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if stop_sequences:
        params["stop_sequences"] = stop_sequences
    if system:
        params["system"] = system
    if tools:
        params["tools"] = tools

    message = get_anthropic_client().messages.create(**params)
    if return_message:
        return message
    return next((b.text for b in message.content if b.type == "text"), "No text response found")

def clean_json(text):
    cleaned_text = text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]
    elif cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[3:]

    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]

    cleaned_text = cleaned_text.strip()

    # 2. Parse into a Python dictionary
    try:
        event_rule = json.loads(cleaned_text)
        print("Success! Parsed dictionary:")
        print(event_rule)
    except json.JSONDecodeError as e:
        print(f"JSON Error: {e}")
        return None
    return event_rule
    
def generate_dataset(model):
    prompt = """
    Generate an evaluation dataset for a prompt evaluation. The dataset will be used to evaluate prompts that generate Python, JSON, or Regex specifically for AWS-related tasks. Generate an array of JSON objects, each representing task that requires Python, JSON, or a Regex to complete.

    Example output:
    ```json
    [
    {
        "task": "Description of task",
    },
    ...additional
    ]
    ```

    * Focus on tasks that can be solved by writing a single Python function, a single JSON object, or a single regex
    * Focus on tasks that do not require writing much code

    Please generate 3 objects.
"""
    messages = []
    add_user_message(messages, prompt)
    text = chat(messages, model = model)
    return clean_json(text)

def run_prompt(test_case):
    """Merges the prompt and test case input, then returns the result"""
    prompt = f"""
    Please solve the following task:

    {test_case["task"]}
    """

    messages = []
    add_user_message(messages, prompt)
    output = chat(messages, model = model)
    return output

def run_test_case(test_case):
    """Calls run_prompt, then grades the result"""
    output = run_prompt(test_case)
    
    # Grading
    model_grade = grade_by_model(test_case, output)
    if model_grade is None:
        return {"output": output, "test_case": test_case,
                "score": None, "reasoning": "Grader returned invalid JSON"}
    model_score = model_grade["score"]
    syntax_score = grade_syntax(output, test_case)

    score = (model_score + syntax_score) / 2

    return {"output": output, "test_case": test_case,
            "score": score, "reasoning": model_grade["reasoning"]}
    
def grade_by_model(test_case, output):
    # Create evaluation prompt
    eval_prompt = f"""
    You are an expert code reviewer. Evaluate this AI-generated solution.
    
    Task: {test_case["task"]}
    Solution: {output}

    Respond with ONLY a JSON object, no other text, with:
    - "strengths": An array of 1-3 key strengths
    - "weaknesses": An array of 1-3 key areas for improvement  
    - "reasoning": A concise explanation of your assessment
    - "score": A number between 1-10
    """
    
    messages = []
    add_user_message(messages, eval_prompt)
    eval_text = chat(messages, model = model)
    return clean_json(eval_text)

def validate_json(text):
    try:
        json.loads(text.strip())
        return 10
    except json.JSONDecodeError:
        return 0

def validate_python(text):
    try:
        ast.parse(text.strip())
        return 10
    except SyntaxError:
        return 0

def validate_regex(text):
    try:
        re.compile(text.strip())
        return 10
    except re.error:
        return 0

def grade_syntax(output: str, test_case: dict) -> int:
    # Example logic using the expected format from your test_case
    expected_format = test_case.get("format")
    
    if expected_format == "json":
        return validate_json(output)
    elif expected_format == "python":
        return validate_python(output)
    elif expected_format == "regex":
        return validate_regex(output)
    return 0

def run_eval(dataset):
    results = [run_test_case(tc) for tc in dataset]
    scores = [r["score"] for r in results if r["score"] is not None]
    average_score = mean(scores) if scores else 0
    print(f"Average score: {average_score} ({len(scores)}/{len(results)} graded)")
    return results

def get_current_datetime(date_format="%Y-%m-%d %H:%M:%S"):
    if not date_format:
        raise ValueError("date_format cannot be empty")
    return datetime.now().strftime(date_format)

def add_duration_to_datetime(amount, unit, datetime_str=None, date_format="%Y-%m-%d %H:%M:%S"):
    ALLOWED_UNITS = {"seconds", "minutes", "hours", "days", "weeks"}
    if unit not in ALLOWED_UNITS:
        raise ValueError(f"unit must be one of {sorted(ALLOWED_UNITS)}")
    start = datetime.strptime(datetime_str, date_format) if datetime_str else datetime.now()
    return (start + timedelta(**{unit: amount})).strftime(date_format)


def load_tool_schema(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_tool(tool_name, tool_input):
    if tool_name == "get_current_datetime":
        return get_current_datetime(**tool_input)
    elif tool_name == "add_duration_to_datetime":
        return add_duration_to_datetime(**tool_input)
    
    raise ValueError(f"Unknown tool: {tool_name}")


def run_tools(message):
    results = []
    for block in message.content:
        if block.type != "tool_use":
            continue
        try:
            output = run_tool(block.name, block.input)
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": str(output), "is_error": False})
        except Exception as e:
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": str(e), "is_error": True})
    return results

def run_conversation(messages, tools):
    while True:
        response = chat(messages, model=model, tools=tools, return_message=True)
        add_assistant_message(messages, response.content)
        if response.stop_reason != "tool_use":
            break
        add_user_message(messages, run_tools(response))
    return messages

def text_from_message(message):
    return "\n".join(
        [block.text for block in message.content if block.type == "text"]
    )

def get_text_edit_schema(model):
    if model.startswith("claude-3-7-sonnet"):
        return {
            "type": "text_editor_20250124",
            "name": "str_replace_editor",
        }
    elif model.startswith("claude-3-5-sonnet"):
        return {
            "type": "text_editor_20241022", 
            "name": "str_replace_editor",
        }
