from anthropic import Anthropic

from reacter_openapitools import AnthropicAdapter


def main():
    # Initialize the Anthropic client
    anthropic_client = Anthropic(
        api_key="your-anthropic-key")

    # Initialize the adapter
    # adapter = AnthropicAdapter(
    #     api_key="your-openapitools-key", verbose=True)
    
    adapter = AnthropicAdapter(
        folder_path="../openapitools", verbose=True
    )

    # Create a chatbot
    chatbot = adapter.create_anthropic_chatbot(
        anthropic_client=anthropic_client,
        llm_config={
            "model": "claude-3-7-sonnet-20250219",
            "temperature": 0.7,
            "max_tokens": 4096,
            "system": "You are a helpful assistant with access to various tools."
        }
    )

    # Use the chatbot
    response = chatbot["invoke"]("can u pls generate otp for 98989898981")
    print(response["text"])

    # Reset the conversation
    chatbot["reset_conversation"]()


if __name__ == "__main__":
    main()
