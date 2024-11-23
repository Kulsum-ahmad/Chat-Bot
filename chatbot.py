from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Load the GPT-Neo 2.7B model
model_name = "EleutherAI/gpt-neo-2.7B"
model = GPTNeoForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def generate_response(prompt, model, tokenizer, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=max_length, do_sample=True, top_p=0.95, top_k=60)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def chatbot():
    print("Welcome to the GPT-Neo chatbot! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        prompt = f"Human: {user_input}\nAI:"
        response = generate_response(prompt, model, tokenizer)
        print(f"AI: {response}")

if __name__ == "__main__":
    chatbot()