from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7b-Chat-GGUF", model_file="llama-2-7b-chat.Q5_K_M.gguf")

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


def get_prompt(instruction: str) -> str:
    # system = "You are an AI assistant that follows instruction extremely well. Help as much as you can. Give short answers."
    system = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST]"
    print(prompt)
    return prompt


question = "Which city is the capital of India?"

for i in llm(get_prompt(question), stream=True):
    print(i, end="", flush=True)
print()
