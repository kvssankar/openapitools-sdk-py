from openai import OpenAI
from reacter_openapitools import OpenAIAdapter


def main():
    # Initialize the OpenAI client
    openai_client = OpenAI(
        # Replace with your actual OpenAI API key
        api_key="your-openai-key",
    )

    # Initialize the adapter
    adapter = OpenAIAdapter(
        api_key="your-openapitools-key",
        verbose=True
    )

    # Create a chatbot
    chatbot = adapter.create_openai_chatbot(
        openai_client=openai_client,
        llm_config={
            "model": "gpt-4o",  # Or another OpenAI model
            "temperature": 0.7,
            "max_tokens": 4096,
            "system": "You are a helpful assistant with access to various tools."
        }
    )

    # Use the chatbot
    response = chatbot["invoke"]("Can you generate an OTP for 98989898981?")
    print(response["text"])

    # Get conversation history
    history = chatbot["get_conversation_history"]()
    print(f"Conversation has {len(history)} messages")

    # Reset the conversation
    chatbot["reset_conversation"]()


if __name__ == "__main__":
    main()
