from ctransformers import AutoModelForCausalLM
from typing import List

llm = AutoModelForCausalLM.from_pretrained("zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf")

# this model cannot still chat, it is able to complete sentences
# [VIDEO EXAMPLE]
# prompt = "Hi! What is your dog's"
# print(llm(prompt))

# [EXERCISE 1]
# prompt = "the capital of India is" # initial prompt
# MY SOLUTION
# prompt = "The location of India's national government is in the city of"
# print(llm(prompt, max_new_tokens=2))

prompt = "The name of capital of India is"
# print(prompt + llm(prompt))

# PROMPT FORMAT
# Most model cards include a prompt format
# prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"


def get_prompt(instruction: str, history: List[str] = None) -> str:
    # system = "You are an AI assistant that follows instruction extremely well. Help as much as you can. Give short answers."
    system = "You are an AI assistant that gives helpful answers. You answer the question in a short and concise way."
    prompt = f"### System:\n{system}\n\n### User:\n"
    if history is not None:
        prompt += f"This is the conversation history: {''. join(history)}. Now answer the question:"
    prompt += f"{instruction}\n\n### Response:\n"
    print(prompt)
    return prompt


question = "Which city is the capital of India?"

history = []

history.append(question)
answer = ""
for word in llm(get_prompt(question), stream=True):
    print(word, end="", flush=True)
    answer += word
print()

history.append(answer)


question = "And which is of the United States?"

for word in llm(get_prompt(question, history=history), stream=True):
    print(word, end="", flush=True)
print()
