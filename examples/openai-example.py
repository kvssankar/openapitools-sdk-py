# chatbot.py
import json
import time
import sys
from openai import OpenAI
from reacter_openapitools import OpenAIAdapter

# Initialize OpenAI client
openai = OpenAI(
    # Replace with your actual OpenAI API key
    api_key="your-openai-key"
)

# Initialize tools adapter
tools_adapter = OpenAIAdapter(
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

    # Get tools in OpenAI format
    tools = tools_adapter.get_openai_tools()
    print(f"Loaded {len(tools)} tools")

    # Create tool handler
    tool_handler = tools_adapter.create_openai_tool_handler()

    # Set up model configuration
    model_config = {
        "model": "gpt-4o",
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

            # Call OpenAI API
            response = openai.chat.completions.create(**api_options)
            message = response.choices[0].message

            # Clear the thinking indicator
            thinking_indicator = False
            print("\r" + " " * 30 + "\r", end="")

            # Process the response
            if not hasattr(message, "tool_calls") or not message.tool_calls:
                # Simple text response
                print("Assistant: ", end="")
                print_typing_effect(message.content)

                # Add assistant message to history
                messages.append({
                    "role": "assistant",
                    "content": message.content
                })
            else:
                # Message with tool calls
                if message.content:
                    print("Assistant: ", end="")
                    print_typing_effect(message.content)

                # Add the assistant's message with tool calls to history
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [tc.model_dump() for tc in message.tool_calls]
                })

                # Process each tool call
                for tool_call in message.tool_calls:
                    print(
                        f"\nAssistant is using tool: {tool_call.function.name}")
                    try:
                        args = json.loads(tool_call.function.arguments)
                        print(f"Tool input: {json.dumps(args, indent=2)}")
                    except:
                        print(f"Tool input: {tool_call.function.arguments}")

                    # Execute the tool
                    try:
                        tool_result = tool_handler({
                            "id": tool_call.id,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        })

                        # Add tool response to messages
                        if tool_result.error:
                            print(f"\nTool Error: {tool_result.error}")
                            messages.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": tool_call.function.name,
                                "content": json.dumps({"error": tool_result.error})
                            })
                        else:
                            print(f"\nTool Result: {tool_result.output}")
                            messages.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": tool_call.function.name,
                                "content": json.dumps({"output": tool_result.output})
                            })

                    except Exception as e:
                        print(f"\nTool execution error: {str(e)}")
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": json.dumps({"error": f"Failed to execute tool: {str(e)}"})
                        })

                # Get continuation from AI after all tool uses
                continuation_options = {
                    "messages": messages,
                    **model_config
                }

                # Add tools if available
                if tools and len(tools) > 0:
                    continuation_options["tools"] = tools

                # Get continuation
                continuation = openai.chat.completions.create(
                    **continuation_options)
                continuation_message = continuation.choices[0].message

                # Check if the continuation has more tool calls
                if hasattr(continuation_message, "tool_calls") and continuation_message.tool_calls:
                    print("\nAssistant needs to use more tools...")
                    # Add the message back to history and let the next loop iteration handle it
                    messages.append({
                        "role": "assistant",
                        "content": continuation_message.content,
                        "tool_calls": [tc.model_dump() for tc in continuation_message.tool_calls]
                    })

                    # Continue immediately with this message
                    response = continuation
                else:
                    # Process continuation text
                    print("\nAssistant: ", end="")
                    print_typing_effect(continuation_message.content)

                    # Add continuation to history
                    messages.append({
                        "role": "assistant",
                        "content": continuation_message.content
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
