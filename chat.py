from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained("zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf")

# this model cannot still chat, it is able to complete sentences
# [VIDEO EXAMPLE]
# prompt = "Hi! What is your dog's"
# print(llm(prompt))

# [EXERCISE 1]
prompt = "The location of India's national government is in the city of"
# prompt = "the capital of India is" # initial prompt
print(llm(prompt, max_new_tokens=2))

# system = "You are an AI assistant that follows instruction extremely well. Help as much as you can. Give short answers."
# instruction = "Which is the biggest city of India?"

# prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
# for i in llm(prompt, stream=True):
#     print(i, end="", flush=True)
# print()
