# chatbot.py
import json
import time
import sys
from anthropic import Anthropic
from reacter_openapitools import AnthropicAdapter

# Initialize Anthropic client
anthropic = Anthropic(
    # Replace with your actual Anthropic API key
    api_key="your-anthropic-key"
)

# Initialize tools adapter
tools_adapter = AnthropicAdapter(
    # Replace with your actual Tools API key
    api_key="your-openapitools-key",
    auto_refresh_count=50,  # Refresh tools after 50 calls
    verbose=True
)


def print_typing_effect(text):
    """Print with a typing effect for assistant messages."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.01)
    print()


def main():
    print("Initializing tools...")
    tools_adapter.initialize()

    # Get tools in Anthropic format
    tools = tools_adapter.get_anthropic_tools()
    print(f"Loaded {len(tools)} tools")

    # Create tool handler
    tool_handler = tools_adapter.create_anthropic_tool_handler()

    # Set up model configuration
    model_config = {
        "model": "claude-3-7-sonnet-20250219",
        "temperature": 0.7,
        "max_tokens": 4096
    }

    # Initialize conversation history
    messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm your AI assistant with tool capabilities. How can I help you today?"
        }
    ]

    # Start conversation
    print("\n=== AI Assistant with Tools ===")
    print("Type 'exit' to quit, 'reset' to start a new conversation")
    print("\nAssistant: ", end="")
    print_typing_effect(messages[0]["content"])

    # Chat loop
    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        if user_input.lower() == "reset":
            messages = [
                {
                    "role": "assistant",
                    "content": "Conversation reset. How can I help you?"
                }
            ]
            print("\nConversation reset.")
            print("\nAssistant: ", end="")
            print_typing_effect(messages[0]["content"])
            continue

        # Add user message to history
        messages.append({
            "role": "user",
            "content": user_input
        })

        # Show "thinking" indicator
        print("Assistant is thinking", end="")
        thinking_indicator = True

        def show_thinking():
            while thinking_indicator:
                sys.stdout.write(".")
                sys.stdout.flush()
                time.sleep(0.5)

        try:
            # Create API call options
            api_options = {
                "messages": messages,
                **model_config
            }

            # Add tools if available
            if tools and len(tools) > 0:
                api_options["tools"] = tools

            # Call Anthropic API
            response = anthropic.messages.create(**api_options)

            # Clear the thinking indicator
            thinking_indicator = False
            print("\r" + " " * 30 + "\r", end="")

            # Process the response
            for content in response.content:
                if content.type == "text":
                    print("Assistant: ", end="")
                    print_typing_effect(content.text)

                    # Add assistant message to history
                    messages.append({
                        "role": "assistant",
                        "content": content.text
                    })

                elif content.type == "tool_use":
                    print(f"\nAssistant is using tool: {content.name}")
                    print(f"Tool input: {json.dumps(content.input, indent=2)}")

                    # Add the assistant's tool use message to history
                    messages.append({
                        "role": "assistant",
                        "content": [{"type": item.type, **item.model_dump()} for item in response.content]
                    })

                    # Execute the tool
                    try:
                        tool_result = tool_handler({
                            "id": content.id,
                            "name": content.name,
                            "input": content.input
                        })

                        # Add tool response to messages
                        if tool_result.error:
                            print(f"\nTool Error: {tool_result.error}")
                            messages.append({
                                "role": "user",
                                "content": [{
                                    "type": "tool_result",
                                    "tool_use_id": content.id,
                                    "content": json.dumps({"error": tool_result.error})
                                }]
                            })
                        else:
                            print(f"\nTool Result: {tool_result.output}")
                            messages.append({
                                "role": "user",
                                "content": [{
                                    "type": "tool_result",
                                    "tool_use_id": content.id,
                                    "content": json.dumps({"output": tool_result.output})
                                }]
                            })

                        # Get continuation from AI after tool use
                        continuation_options = {
                            "messages": messages,
                            **model_config
                        }

                        # Add tools if available
                        if tools and len(tools) > 0:
                            continuation_options["tools"] = tools

                        # Get continuation
                        continuation = anthropic.messages.create(
                            **continuation_options)

                        # Process continuation
                        continuation_text = "".join(
                            item.text for item in continuation.content if item.type == "text"
                        )

                        print("\nAssistant: ", end="")
                        print_typing_effect(continuation_text)

                        # Add continuation to history
                        messages.append({
                            "role": "assistant",
                            "content": continuation_text
                        })

                    except Exception as e:
                        print(f"\nTool execution error: {str(e)}")
                        messages.append({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": json.dumps({"error": f"Failed to execute tool: {str(e)}"})
                            }]
                        })

        except Exception as e:
            # Clear the thinking indicator
            thinking_indicator = False
            print("\r" + " " * 30 + "\r", end="")
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
