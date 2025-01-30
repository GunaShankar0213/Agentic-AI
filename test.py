# import logging
# from transformers import GPT2LMHeadModel, GPT2Tokenizer

# # Initialize logger
# logging.basicConfig(level=logging.INFO)

# # Load the model and tokenizer
# def load_model_and_tokenizer(model_name="gpt2"):
#     """Load GPT-2 model and tokenizer."""
#     try:
#         model = GPT2LMHeadModel.from_pretrained(model_name)
#         tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#         logging.info(f"Model '{model_name}' loaded successfully.")
#         return model, tokenizer
#     except Exception as e:
#         logging.error(f"Error loading model: {str(e)}")
#         return None, None

# # Generate a response using the model
# def generate_response(model, tokenizer, prompt="Hello, how are you?"):
#     """Generate response using the GPT-2 model."""
#     try:
#         inputs = tokenizer.encode(prompt, return_tensors="pt")
#         outputs = model.generate(inputs, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return response
#     except Exception as e:
#         logging.error(f"Error generating response: {str(e)}")
#         return "Error generating response."

# # Main function to run the test
# if __name__ == "__main__":
#     # Load model and tokenizer
#     model, tokenizer = load_model_and_tokenizer("gpt2")  # You can use 'distilgpt2' for smaller size

#     if model and tokenizer:
#         # Test the model with a simple query
#         prompt = "What is the capital of France?"
#         response = generate_response(model, tokenizer, prompt)
#         print("Model Response:", response)
#     else:
#         print("Failed to load the model.")
from ollama._client import Client

# Initialize the client
_ = Client()

# List all available models
models = _.list()

# Print the models available
for model in models.models:
    print(f"Model: {model.model}, Modified At: {model.modified_at}, Size: {model.size}")

