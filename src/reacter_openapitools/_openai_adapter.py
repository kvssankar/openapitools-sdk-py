import json
from typing import Any, Dict, List, Union, Optional

from ._base_adapter import BaseToolsAdapter, Tool, ToolExecutionResult


class OpenAIAdapter(BaseToolsAdapter):
    """Adapter for OpenAI API."""

    def get_openai_tools(self, tool_names=None) -> List[Dict[str, Any]]:
        """
        Convert tools to OpenAI format.

        Args:
            tool_names: Optional list of specific tools to convert

        Returns:
            Array of tools in OpenAI format
        """
        self.ensure_initialized()

        self.log("Converting tools to OpenAI format...")
        selected_tools = self.get_tools_by_names(tool_names)

        openai_tools = []
        for tool in selected_tools.values():
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema
                }
            })

        self.log(f"Converted {len(openai_tools)} tools to OpenAI format")

        return openai_tools

    def create_openai_tool_handler(self, tool_names=None):
        """
        Creates a function to handle OpenAI tool calls.

        Args:
            tool_names: Optional list of specific tools to handle

        Returns:
            Tool handler function for OpenAI
        """
        self.ensure_initialized()

        self.log("Creating OpenAI tool handler...")
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
            f"OpenAI tool handler created for {len(executors)} tools", True)

        def tool_handler(tool_use: Dict[str, Any]) -> ToolExecutionResult:
            """Handle a tool use request from OpenAI."""
            function_data = tool_use.get("function", {})
            name = function_data.get("name")
            args_str = function_data.get("arguments", "{}")
            self.log(f"OpenAI requested tool: {name}")

            # Parse the arguments string to an object
            try:
                input_args = json.loads(args_str)
                self.log(f"Tool inputs: {json.dumps(input_args)}")
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse tool arguments: {str(e)}"
                self.log_error(error_msg)
                return ToolExecutionResult(error=error_msg)

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

    def create_openai_chatbot(self,
                              openai_client: Any,
                              llm_config: Dict[str, Any] = None,
                              options: Dict[str, Any] = None):
        """
        Creates an OpenAI chatbot with tool capabilities.

        Args:
            openai_client: Initialized OpenAI client
            llm_config: Configuration for the LLM model
            options: Optional additional settings

        Returns:
            A chatbot object with invoke, reset_conversation, and get_conversation_history methods
        """
        llm_config = llm_config or {}
        options = options or {}
        tool_names = options.get("tool_names")

        self.ensure_initialized()

        self.log("Creating OpenAI chatbot...", True)

        # Get tools
        tools = self.get_openai_tools(tool_names)
        self.log(f"Chatbot initialized with {len(tools)} tools", True)

        # Create tool handler
        tool_handler = self.create_openai_tool_handler(tool_names)

        # Default options
        default_llm_config = {
            "model": "gpt-4o",
            "temperature": 0.7,
        }
        default_llm_config.update(llm_config)
        
        # removve system from default_llm_config
        if "system" in default_llm_config:
            del default_llm_config["system"]

        self.log(
            f"Chatbot configured with model: {default_llm_config['model']}", True)

        # Initialize conversation history
        messages = []

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

            # Add system message at the beginning if provided
            if (llm_config.get("system") and
                    len(current_messages) > 0 and
                    current_messages[0].get("role") != "system"):
                current_messages.insert(0, {
                    "role": "system",
                    "content": llm_config.get("system")
                })

            try:
                # Create API call options
                api_options = {
                    "messages": current_messages,
                    **default_llm_config
                }
                
                # Add tools if provided
                if tools and len(tools) > 0:
                    api_options["tools"] = tools

                self.log(f"Calling OpenAI API ({len(tools)} tools enabled)")

                # Call OpenAI API with all options
                response = openai_client.chat.completions.create(**api_options)

                self.log("Received response from OpenAI API")

                # Process the response and handle any tool usage
                result = process_response(response)

                return {
                    "text": result,
                    "messages": messages.copy()
                }
            except Exception as e:
                # Handle errors with more details
                error_message = f"Error in OpenAI API: {str(e)}"

                if hasattr(e, "status"):
                    error_message = f"OpenAI API Error ({e.status}): {str(e)}"
                    if hasattr(e, "error"):
                        error_message += f"\nDetails: {json.dumps(e.error)}"

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
            Process the response from OpenAI, handling any tool usage.

            Args:
                response: The response from OpenAI

            Returns:
                The final text response
            """
            message = response.choices[0].message

            # Check if there are tool calls
            if hasattr(message, "tool_calls") and message.tool_calls:
                tool_calls = message.tool_calls
                self.log(f"Detected {len(tool_calls)} tool call(s)", True)

                # Add the assistant's message with tool calls to history
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [tc.model_dump() for tc in tool_calls]
                })

                # Process each tool call
                tool_results = []
                for tool_call in tool_calls:
                    self.log(f"Executing tool call: {tool_call.function.name}", True)

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
                        content = (json.dumps({"error": tool_result.error}) 
                                  if tool_result.error 
                                  else json.dumps({"output": tool_result.output}))
                        
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": content
                        })
                    except Exception as e:
                        self.log_error(f"Tool handler error: {str(e)}")
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": json.dumps({
                                "error": f"Failed to execute tool: {str(e)}"
                            })
                        })

                # Add all tool results to messages
                messages.extend(tool_results)

                # Create API call options for continuation
                api_options = {
                    "messages": messages.copy(),
                    **default_llm_config
                }

                # Add tools again for potential future tool calls
                if tools and len(tools) > 0:
                    api_options["tools"] = tools

                self.log("Requesting continuation from OpenAI after tool use")

                # Get continuation from AI after tool use
                continuation = openai_client.chat.completions.create(**api_options)

                # Check if the continuation contains more tool calls
                continuation_message = continuation.choices[0].message
                if (hasattr(continuation_message, "tool_calls") and 
                        continuation_message.tool_calls):
                    # Recursive call to handle nested tool usage
                    self.log("Detected nested tool usage, processing recursively", True)
                    nested_result = process_response(continuation)
                    return nested_result
                else:
                    # Add continuation to history
                    messages.append({
                        "role": "assistant",
                        "content": continuation_message.content
                    })

                    self.log(f"Received continuation text ({len(continuation_message.content)} chars)")
                    return continuation_message.content
            else:
                # Simple text response without tool usage
                response_text = message.content or ""

                # Add the response to conversation history
                messages.append({
                    "role": "assistant",
                    "content": response_text
                })

                self.log(f"Received text-only response ({len(response_text)} chars)")
                return response_text

        def reset_conversation():
            """Resets the conversation history."""
            self.log("Conversation history reset", True)
            messages.clear()

        def get_conversation_history():
            """Gets the current conversation history."""
            self.log(f"Retrieved conversation history ({len(messages)} messages)")
            return messages.copy()

        # Return the chatbot interface
        return {
            "invoke": invoke,
            "reset_conversation": reset_conversation,
            "get_conversation_history": get_conversation_history
        }