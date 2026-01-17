# prompt_engineering.py
# Explanation of Prompt Engineering
# Prompt engineering is the practice of crafting effective prompts to get desired outputs from LLMs.
# It involves structuring questions, providing context, and using techniques like few-shot learning.
# Good prompts can dramatically improve LLM performance and consistency.

# Simple examples of different prompting techniques

# Zero-shot prompting function - ask model to perform task without examples
def zero_shot_prompting(llm_function, task_description, input_text):
    """
    Zero-shot prompting: Ask the model to perform a task without examples.
    This relies on the model's pre-trained knowledge and general capabilities.
    Works well for tasks the model has seen during training.
    """
    # Construct prompt with clear task description and input
    prompt = f"{task_description}\n\nInput: {input_text}\nOutput:"
    # Call the LLM function with the constructed prompt
    return llm_function(prompt)

# Few-shot prompting function - provide examples before the actual task
def few_shot_prompting(llm_function, examples, task_description, input_text):
    """
    Few-shot prompting: Provide examples before asking for the task.
    This helps the model understand the desired format and reasoning pattern.
    Particularly effective for tasks requiring specific output formats.
    """
    # Start with the task description
    prompt = task_description + "\n\n"
    # Add each example in a consistent format
    for example_input, example_output in examples:
        prompt += f"Input: {example_input}\nOutput: {example_output}\n\n"
    # Add the actual input to be processed
    prompt += f"Input: {input_text}\nOutput:"
    # Call the LLM function with the example-rich prompt
    return llm_function(prompt)

# Mock LLM function for demonstration purposes
# Simulates how a real LLM would respond to different prompts
def mock_llm(prompt):
    # Check if the prompt is about sentiment analysis
    if "sentiment" in prompt.lower():
        # Simple keyword-based sentiment detection
        if "happy" in prompt.lower() or "good" in prompt.lower():
            return "Positive"
        elif "sad" in prompt.lower() or "bad" in prompt.lower():
            return "Negative"
        else:
            return "Neutral"
    else:
        # Default response for other types of prompts
        return "This is a generated response based on the prompt."

# Example usage demonstrating different prompting techniques
task = "Classify the sentiment of the following text as Positive, Negative, or Neutral."

# Zero-shot prompting example
input_text = "I love this product!"
zero_shot_result = zero_shot_prompting(mock_llm, task, input_text)
print(f"Zero-shot result: {zero_shot_result}")

# Few-shot prompting example with training examples
examples = [
    ("This is great!", "Positive"),    # Example of positive sentiment
    ("I hate this.", "Negative")       # Example of negative sentiment
]
few_shot_result = few_shot_prompting(mock_llm, examples, task, input_text)
print(f"Few-shot result: {few_shot_result}")