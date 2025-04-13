from typing import Any, Dict, List, Optional, Callable, Union
import inspect
import json

from ._base_adapter import BaseToolsAdapter, Tool, ToolExecutionResult


class LangChainAdapter(BaseToolsAdapter):
    """Adapter for LangChain tools."""

    def convert_to_langchain_tool(self, tool: Tool) -> Any:
        """
        Convert a Tool instance to a LangChain tool.

        Args:
            tool: Tool instance to convert

        Returns:
            LangChain tool instance
        """
        try:
            # Import LangChain components
            from langchain.tools import StructuredTool
            from langchain.pydantic_v1 import create_model, Field

            self.log(
                f"Converting tool '{tool.name}' to LangChain StructuredTool")

            # Create dynamic Pydantic model for input schema
            properties = tool.input_schema.get("properties", {})
            required = tool.input_schema.get("required", [])

            # Build field definitions for the input model
            fields = {}
            for name, schema in properties.items():
                field_type = self._map_json_schema_to_python_type(
                    schema.get("type", "string"))
                description = schema.get("description", "")

                # Check if field is required
                is_required = name in required

                # Create field with appropriate annotation
                if is_required:
                    fields[name] = (field_type, Field(description=description))
                else:
                    fields[name] = (Optional[field_type], Field(
                        None, description=description))

            # Create Pydantic model for tool input
            input_model = create_model(f"{tool.name}Input", **fields)

            # Create handler function that will execute the tool
            def tool_handler(**kwargs):
                # Execute the tool with the input args
                if tool.script_type == "python":
                    result = self.execute_python_tool(tool, kwargs)
                elif tool.script_type == "bash":
                    result = self.execute_bash_tool(tool, kwargs)
                else:
                    return f"Error: Unsupported script type: {tool.script_type}"

                # Return the result
                if result.error:
                    return f"Error: {result.error}"
                return result.output

            # Update function signature metadata for better integration
            tool_handler.__name__ = tool.name
            tool_handler.__doc__ = tool.description

            # Create and return StructuredTool
            return StructuredTool.from_function(
                func=tool_handler,
                name=tool.name,
                description=tool.description,
                args_schema=input_model,
                return_direct=True
            )

        except ImportError as e:
            self.log_error(f"Failed to import LangChain: {str(e)}")
            raise ImportError(
                "LangChain is required. Install with: pip install langchain")
        except Exception as e:
            self.log_error(
                f"Failed to convert tool to LangChain format: {str(e)}")
            raise

    def _map_json_schema_to_python_type(self, json_type: str) -> type:
        """
        Map JSON schema type to Python type.

        Args:
            json_type: JSON schema type

        Returns:
            Corresponding Python type
        """
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict
        }

        return type_map.get(json_type, str)

    def get_langchain_tools(self, tool_names=None) -> List[Any]:
        """
        Convert tools to LangChain format.

        Args:
            tool_names: Optional list of specific tools to convert

        Returns:
            List of LangChain tools
        """
        self.ensure_initialized()

        self.log("Converting tools to LangChain format...")
        selected_tools = self.get_tools_by_names(tool_names)

        langchain_tools = []
        for tool in selected_tools.values():
            try:
                langchain_tool = self.convert_to_langchain_tool(tool)
                langchain_tools.append(langchain_tool)
                self.log(f"Converted tool: {tool.name}")
            except Exception as e:
                self.log_error(
                    f"Failed to convert tool '{tool.name}': {str(e)}")

        self.log(f"Converted {len(langchain_tools)} tools to LangChain format")
        return langchain_tools

    def create_structured_tool_from_function(self, func: Callable, name: str = None,
                                             description: str = None) -> Any:
        """
        Create a LangChain StructuredTool from a Python function.

        Args:
            func: Function to convert to a tool
            name: Optional name for the tool (defaults to function name)
            description: Optional description (defaults to function docstring)

        Returns:
            LangChain StructuredTool instance
        """
        try:
            # Import LangChain components
            from langchain.tools import StructuredTool

            self.log(f"Creating StructuredTool from function: {func.__name__}")

            # Use function name if name not provided
            tool_name = name or func.__name__

            # Use docstring if description not provided
            tool_description = description or func.__doc__ or f"Tool for {tool_name}"

            # Create and return the StructuredTool
            structured_tool = StructuredTool.from_function(
                func=func,
                name=tool_name,
                description=tool_description
            )

            self.log(f"Created StructuredTool: {tool_name}")
            return structured_tool

        except ImportError:
            self.log_error("Failed to import LangChain")
            raise ImportError(
                "LangChain is required. Install with: pip install langchain")
        except Exception as e:
            self.log_error(f"Failed to create StructuredTool: {str(e)}")
            raise
