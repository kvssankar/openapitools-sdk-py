import json
from typing import Any, Dict, List, Union

from ._base_adapter import BaseToolsAdapter, Tool, ToolExecutionResult


class AnthropicAdapter(BaseToolsAdapter):
    """Adapter for Anthropic Claude API."""

    def get_anthropic_tools(self, tool_names=None) -> List[Dict[str, Any]]:
        """
        Convert tools to Anthropic format.

        Args:
            tool_names: Optional list of specific tools to convert

        Returns:
            Array of tools in Anthropic format
        """
        self.ensure_initialized()

        self.log("Converting tools to Anthropic format...")
        selected_tools = self.get_tools_by_names(tool_names)

        anthropic_tools = []
        for tool in selected_tools.values():
            anthropic_tools.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema
            })

        self.log(f"Converted {len(anthropic_tools)} tools to Anthropic format")

        return anthropic_tools

    def create_anthropic_tool_handler(self, tool_names=None):
        """
        Creates a function to handle Anthropic tool calls.

        Args:
            tool_names: Optional list of specific tools to handle

        Returns:
            Tool handler function for Anthropic
        """
        self.ensure_initialized()

        self.log("Creating Anthropic tool handler...")
        selected_tools = self.get_tools_by_names(tool_names)
        executors = {}

        # Create a dictionary of tool executors
        for tool_name, tool in selected_tools.items():
            # Create a tool executor for each tool
            def tool_executor(args, tool=tool):
                # Check if we should execute Python or bash
                if tool.script_type == "python":
                    return self.execute_python_tool(tool, args)
                elif tool.script_type == "bash":
                    return self.execute_bash_tool(tool, args)
                else:
                    return ToolExecutionResult(
                        error=f"Unsupported script type: {tool.script_type}"
                    )

            executors[tool.name] = tool_executor
            self.log(f"Created executor for tool: {tool.name}")

        self.log(
            f"Anthropic tool handler created for {len(executors)} tools", True)

        def tool_handler(tool_use: Dict[str, Any]) -> ToolExecutionResult:
            """Handle a tool use request from Anthropic."""
            name = tool_use.get("name")
            input_args = tool_use.get("input", {})
            self.log(f"Anthropic requested tool: {name}")

            self.log(f"Tool inputs: {json.dumps(input_args)}")

            if name in executors:
                self.log(f"Executing tool: {name}")
                result = executors[name](input_args)
                if result.error:
                    self.log_error(
                        f"Tool {name} execution failed: {result.error}")
                    return ToolExecutionResult(
                        error=f"Something went wrong with the tool execution. Details: {result.error}"
                    )

                self.log(f"Tool {name} output: {result.output}")
                self.log(f"Tool {name} executed successfully")
                return result
            else:
                error_msg = f"Tool {name} not found in available tools"
                self.log_error(error_msg)
                return ToolExecutionResult(error=error_msg)

        return tool_handler

    def create_anthropic_chatbot(self,
                                 anthropic_client: Any,
                                 llm_config: Dict[str, Any] = None,
                                 options: Dict[str, Any] = None):
        """
        Creates an Anthropic chatbot with tool capabilities.

        Args:
            anthropic_client: Initialized Anthropic client
            llm_config: Configuration for the LLM model
            options: Optional additional settings

        Returns:
            A chatbot object with invoke, reset_conversation, and get_conversation_history methods
        """
        llm_config = llm_config or {}
        options = options or {}
        tool_names = options.get("tool_names")

        self.ensure_initialized()

        self.log("Creating Anthropic chatbot...", True)

        # Get tools
        tools = self.get_anthropic_tools(tool_names)
        self.log(f"Chatbot initialized with {len(tools)} tools", True)

        # Create tool handler
        tool_handler = self.create_anthropic_tool_handler(tool_names)

        # Default options
        default_llm_config = {
            "model": "claude-3-7-sonnet-20250219",
            "temperature": 0.7
        }
        default_llm_config.update(llm_config)

        self.log(
            f"Chatbot configured with model: {default_llm_config['model']}", True)

        # Initialize conversation history
        messages = options.get("messages", [])

        def invoke(user_input: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
            """
            Invokes the chatbot with a user message.

            Args:
                user_input: The message from the user (string or complex message object)

            Returns:
                The chatbot's response
            """
            self.log("Processing user message")

            # Handle different input types
            if isinstance(user_input, str):
                # Simple text message
                user_message = {
                    "role": "user",
                    "content": user_input
                }
            else:
                # Complex message object
                user_message = {
                    "role": "user",
                    "content": user_input
                }

            # Add user message to history
            messages.append(user_message)

            # Create a copy of messages for this invocation
            current_messages = messages.copy()

            try:
                # Create API call options
                api_options = {
                    "messages": current_messages,
                    **default_llm_config
                }

                # Add tools if provided
                if tools and len(tools) > 0:
                    api_options["tools"] = tools

                self.log(f"Calling Anthropic API ({len(tools)} tools enabled)")

                # Call Anthropic API with all options
                response = anthropic_client.messages.create(**api_options)

                self.log("Received response from Anthropic API")

                # Process the response and handle any tool usage
                result = process_response(response)

                return {
                    "text": result,
                    "messages": messages.copy()
                }
            except Exception as e:
                # Handle errors with more details
                error_message = f"Error in Anthropic API: {str(e)}"

                if hasattr(e, "status_code"):
                    error_message = f"Anthropic API Error ({e.status_code}): {str(e)}"

                self.log_error(error_message)

                messages.append({
                    "role": "assistant",
                    "content": error_message
                })

                return {
                    "text": error_message,
                    "messages": messages.copy()
                }

        def process_response(response) -> str:
            """
            Process the response from Anthropic, handling any tool usage.

            Args:
                response: The response from Anthropic

            Returns:
                The final text response
            """
            # Full assistant response text
            response_text = ""

            # Check if response contains only text
            if all(item.type == "text" for item in response.content):
                # If response contains only text, add it directly
                full_text = "".join(
                    item.text for item in response.content if item.type == "text")

                messages.append({
                    "role": "assistant",
                    "content": full_text
                })

                self.log(
                    f"Received text-only response ({len(full_text)} chars)")
                return full_text

            self.log("Processing complex response with multiple content types")

            # Process each content item in the response
            for content in response.content:
                if content.type == "text":
                    response_text += content.text
                    self.log(f"Added text content ({len(content.text)} chars)")

                elif content.type == "tool_use" and tool_handler:
                    # If there's tool usage, we'll need to execute the tool and get continuation
                    self.log(
                        f"Detected tool use request: {content.name}", True)

                    # First, add the assistant's tool use message to history
                    messages.append({
                        "role": "assistant",
                        "content": [{"type": item.type, **item.model_dump()} for item in response.content]
                    })

                    # Execute the tool
                    try:
                        self.log(f"Executing tool {content.name}...")

                        tool_result = tool_handler({
                            "id": content.id,
                            "name": content.name,
                            "input": content.input
                        })

                        # Add tool response to messages following Anthropic's expected format
                        if tool_result.error:
                            self.log_error(
                                f"Tool execution error for {content.name}: {tool_result.error}")
                            messages.append({
                                "role": "user",
                                "content": [{
                                    "type": "tool_result",
                                    "tool_use_id": content.id,
                                    "content": json.dumps({"error": tool_result.error})
                                }]
                            })
                        else:
                            self.log(
                                f"Tool {content.name} executed successfully")
                            self.log(f"Tool result: {tool_result.output}")

                            messages.append({
                                "role": "user",
                                "content": [{
                                    "type": "tool_result",
                                    "tool_use_id": content.id,
                                    "content": json.dumps({"output": tool_result.output})
                                }]
                            })

                    except Exception as e:
                        self.log_error(f"Tool handler error: {str(e)}")
                        messages.append({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": json.dumps({
                                    "error": f"Failed to execute tool: {str(e)}"
                                })
                            }]
                        })

                    # Create API call options for continuation
                    api_options = {
                        "messages": messages,
                        **default_llm_config
                    }

                    # Add tools if provided
                    if tools and len(tools) > 0:
                        api_options["tools"] = tools

                    self.log(
                        "Requesting continuation from Anthropic after tool use")

                    # Get continuation from AI after tool use
                    continuation = anthropic_client.messages.create(
                        **api_options)

                    # Check if the continuation contains more tool usage
                    if any(item.type == "tool_use" for item in continuation.content):
                        # Recursive call to handle nested tool usage
                        self.log(
                            "Detected nested tool usage, processing recursively", True)
                        nested_result = process_response(continuation)
                        response_text += nested_result
                    else:
                        # Add the continuation text
                        continuation_text = "".join(
                            item.text for item in continuation.content if item.type == "text"
                        )

                        response_text += continuation_text
                        self.log(
                            f"Received continuation text ({len(continuation_text)} chars)")

                        # Add continuation to history
                        messages.append({
                            "role": "assistant",
                            "content": continuation_text
                        })

            return response_text

        def reset_conversation():
            """Resets the conversation history."""
            self.log("Conversation history reset", True)
            messages.clear()

        def get_conversation_history():
            """Gets the current conversation history."""
            self.log(
                f"Retrieved conversation history ({len(messages)} messages)")
            return messages.copy()

        # Return the chatbot interface
        return {
            "invoke": invoke,
            "reset_conversation": reset_conversation,
            "get_conversation_history": get_conversation_history
        }
