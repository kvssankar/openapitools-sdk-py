import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Union, Set
import requests
import io
from contextlib import redirect_stdout, redirect_stderr


class ToolExecutionResult:
    """Tool execution result."""

    def __init__(self, output: Optional[str] = None, error: Optional[str] = None):
        self.output = output
        self.error = error

    def to_dict(self) -> Dict[str, Optional[str]]:
        """Convert to dictionary representation."""
        return {
            "output": self.output,
            "error": self.error
        }


class EnvironmentCheck:
    """Environment check result."""

    def __init__(self, script_type: str, valid: bool, executor: str, error: Optional[str] = None):
        self.script_type = script_type
        self.valid = valid
        self.executor = executor
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "scriptType": self.script_type,
            "valid": self.valid,
            "executor": self.executor,
        }
        if self.error:
            result["error"] = self.error
        return result


class Tool:
    """Represents a tool with execution capabilities."""

    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        script: str,
        script_type: str,
        version_name: str,
        script_path: Optional[str] = None  # Add script_path for local mode
    ):
        self.id = id
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.script = script
        self.script_type = script_type
        self.version_name = version_name
        self.script_path = script_path  # Path to script file for local mode

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "script": self.script,
            "script_type": self.script_type,
            "version_name": self.version_name
        }
        if self.script_path:
            result["script_path"] = self.script_path
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Tool':
        """Create a Tool instance from a dictionary."""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            input_schema=data.get("input_schema", {}),
            script=data.get("script", ""),
            script_type=data.get("script_type", "bash"),
            version_name=data.get("version_name", ""),
            script_path=data.get("script_path")
        )


class BaseToolsAdapter:
    """Base adapter class for tools."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        folder_path: Optional[str] = None,
        auto_refresh_count: int = 100,
        skip_environment_check: bool = False,
        verbose: bool = True
    ):
        """
        Initialize the tools adapter.

        Args:
            api_key: API key for authentication (required for API mode)
            api_url: API URL for fetching tools (used in API mode)
            folder_path: Path to folder containing tool files (for local mode)
            auto_refresh_count: Number of tool calls before auto-refresh
            skip_environment_check: Skip initial environment check
            verbose: Enable verbose logging
        """
        self.api_key = api_key
        self.folder_path = folder_path
        self.tools_map: Dict[str, Tool] = {}
        self.initialized = False
        self.api_url = api_url or "https://s8ka4ekkbp.us-east-1.awsapprunner.com"
        self.environment_checks: Dict[str, EnvironmentCheck] = {}
        self.tool_call_count = 0
        self.auto_refresh_count = auto_refresh_count
        self.skip_initial_environment_check = skip_environment_check
        self.environment_variables: Dict[str, str] = {}
        self.verbose = verbose

        # Determine if we're using local files or API
        self.local_mode = folder_path is not None

        if not self.local_mode and not self.api_key:
            raise ValueError("Either api_key or folder_path must be provided")

    def log(self, message: str, force: bool = False) -> None:
        """
        Log message based on verbose setting.

        Args:
            message: Message to log
            force: Force log regardless of verbose setting
        """
        if self.verbose:
            print(f"[OpenAPI Tools SDK] {message}")

    def log_error(self, message: str) -> None:
        """
        Log error message (always shown).

        Args:
            message: Error message to log
        """
        print(f"[OpenAPI Tools SDK ERROR] {message}", file=sys.stderr)

    def check_environment(self, script_type: str) -> EnvironmentCheck:
        """
        Check if the current OS supports the script type.

        Args:
            script_type: Type of script (bash or python)

        Returns:
            Environment check result
        """
        # If we already checked this script type, return cached result
        if script_type in self.environment_checks:
            return self.environment_checks[script_type]

        result = None

        if script_type == "python":
            # Check Python availability - this is simpler in Python since we're already running Python
            result = EnvironmentCheck(
                script_type=script_type,
                valid=True,
                executor=sys.executable
            )

        elif script_type == "bash":
            # Check bash availability
            try:
                proc = subprocess.run(
                    ["bash", "--version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False
                )

                if proc.returncode == 0:
                    result = EnvironmentCheck(
                        script_type=script_type,
                        valid=True,
                        executor="bash"
                    )
                else:
                    result = EnvironmentCheck(
                        script_type=script_type,
                        valid=False,
                        executor="",
                        error="Bash is not installed or not available in PATH."
                    )
            except Exception as e:
                result = EnvironmentCheck(
                    script_type=script_type,
                    valid=False,
                    executor="",
                    error=f"Error checking bash: {str(e)}"
                )
        else:
            result = EnvironmentCheck(
                script_type=script_type,
                valid=False,
                executor="",
                error=f"Unsupported script type: {script_type}"
            )

        # Cache the result
        self.environment_checks[script_type] = result
        return result

    def check_all_environments(self) -> Dict[str, EnvironmentCheck]:
        """
        Check the environment for all supported script types.

        Returns:
            Map of environment check results by script type
        """
        # Check all commonly used script types
        self.check_environment("python")
        self.check_environment("bash")

        return self.environment_checks

    def recheck_environment(self, force_refresh: bool = True) -> Dict[str, EnvironmentCheck]:
        """
        Manually recheck the environment.

        Args:
            force_refresh: Force a refresh even if already checked

        Returns:
            Map of environment check results by script type
        """
        if force_refresh:
            # Clear cached results
            self.environment_checks = {}

        # Check all script types used by the current tools
        script_types: Set[str] = set()
        for tool in self.tools_map.values():
            script_types.add(tool.script_type)

        # Always check the common script types anyway
        script_types.add("python")
        script_types.add("bash")

        # Check each script type
        for script_type in script_types:
            self.check_environment(script_type)

        return self.environment_checks

    def initialize(self) -> None:
        """Initializes the adapter by fetching tools from the API or local folder."""
        if self.initialized:
            return

        try:
            self.log("Initializing tools adapter...", True)

            if self.local_mode:
                self.load_tools_from_folder()
            else:
                self.fetch_tools()

            # Perform environment checks for all tool script types
            if not self.skip_initial_environment_check:
                self.recheck_environment()
                self.log_environment_status()

            self.initialized = True
            self.log(
                f"Initialization complete. {len(self.tools_map)} tools available.", True)
        except Exception as e:
            self.log_error(f"Failed to initialize tools: {str(e)}")
            raise

    def load_tools_from_folder(self) -> None:
        """Loads tools from the specified folder path without loading script content."""
        try:
            self.log(f"Loading tools from folder: {self.folder_path}")

            # Validate folder path
            if not os.path.exists(self.folder_path):
                raise FileNotFoundError(
                    f"Folder path does not exist: {self.folder_path}")

            # Load tools.json file
            tools_file_path = os.path.join(self.folder_path, "tools.json")
            if not os.path.exists(tools_file_path):
                raise FileNotFoundError(
                    f"tools.json file not found in {self.folder_path}")

            with open(tools_file_path, 'r') as f:
                tools_data = json.load(f)

            # Process each tool
            for tool_data in tools_data:
                tool_name = tool_data["name"]
                tool_id = tool_data.get("id", "")
                production_version_name = tool_data.get(
                    "production_version_name", "")

                # Access the production version from the versions map
                versions = tool_data.get("versions", {})

                # Check if versions is a dictionary
                if not isinstance(versions, dict):
                    self.log_error(
                        f"Versions for tool {tool_name} is not a dictionary")
                    continue

                # Get the production version from the versions map
                if production_version_name not in versions:
                    self.log_error(
                        f"Production version {production_version_name} not found for tool {tool_name}")
                    continue

                production_version = versions[production_version_name]

                # Create script path instead of loading content
                script_extension = ".py" if production_version.get(
                    "script_type") == "python" else ".sh"
                script_filename = f"{tool_name}-{production_version_name}{script_extension}"
                script_path = os.path.join(self.folder_path, script_filename)

                if not os.path.exists(script_path):
                    self.log_error(f"Script file not found: {script_path}")
                    continue

                # Create the tool object without loading script content
                self.tools_map[tool_name.lower()] = Tool(
                    id=tool_id,
                    name=tool_name,
                    description=production_version.get("description", ""),
                    input_schema=production_version.get("input_schema", {}),
                    script="",  # Empty script content
                    script_type=production_version.get("script_type", "bash"),
                    version_name=production_version_name,
                    script_path=script_path  # Store the path instead
                )

                self.log(
                    f"Loaded tool reference: {tool_name} (version: {production_version_name})")

        except Exception as e:
            self.log_error(f"Failed to load tools from folder: {str(e)}")
            raise

    def fetch_tools(self) -> None:
        """Fetches tools from the API."""
        try:
            self.log("Fetching tools from API...")

            response = requests.get(
                f"{self.api_url}/api/get-tools",
                headers={"x-api-key": self.api_key}
            )

            if response.status_code != 200:
                raise Exception(
                    f"API request failed with status {response.status_code}")

            result = response.json()
            tools = result.get("data", [])

            # Convert array to map with name as key
            for tool_data in tools:
                name = tool_data["name"].lower()
                self.tools_map[name] = Tool(
                    id=tool_data.get("id", ""),
                    name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    input_schema=tool_data.get("input_schema", {}),
                    script=tool_data.get("script", ""),
                    script_type=tool_data.get("script_type", "bash"),
                    version_name=tool_data.get("version_name", "")
                )
        except Exception as e:
            self.log_error(f"Failed to fetch tools: {str(e)}")
            raise

    def refresh_tools(self) -> None:
        """Manually refresh the tools from the API or local folder."""
        try:
            self.log("Refreshing tools...", True)

            if self.local_mode:
                self.load_tools_from_folder()
            else:
                self.fetch_tools()

            self.tool_call_count = 0  # Reset the counter after refresh
            self.log(
                f"Tools refreshed successfully. {len(self.tools_map)} tools available.", True)
        except Exception as e:
            self.log_error(f"Failed to refresh tools: {str(e)}")
            raise

    def log_environment_status(self) -> None:
        """Log the environment status to the console."""
        self.log("=== Environment Status ===", True)
        for check in self.environment_checks.values():
            if check.valid:
                self.log(
                    f"✅ {check.script_type}: Available (using {check.executor})", True)
            else:
                self.log(
                    f"❌ {check.script_type}: Not available - {check.error}", True)
        self.log("========================", True)

    def check_auto_refresh(self) -> None:
        """Check if auto-refresh is needed and perform if necessary."""
        self.tool_call_count += 1

        if self.auto_refresh_count > 0 and self.tool_call_count >= self.auto_refresh_count:
            self.log(
                f"Auto-refreshing tools after {self.tool_call_count} tool calls", True)
            self.refresh_tools()

    def ensure_initialized(self) -> None:
        """Ensures the adapter is initialized."""
        if not self.initialized:
            self.initialize()

    def set_environment_variables(self, variables: Dict[str, str]) -> None:
        """
        Sets environment variables for tool execution.

        Args:
            variables: Object containing environment variables to set
        """
        self.environment_variables = variables
        self.log(
            f"Set {len(variables)} environment variables for tool execution")

    def add_environment_variable(self, name: str, value: str) -> None:
        """
        Adds an environment variable for tool execution.

        Args:
            name: Name of the environment variable
            value: Value of the environment variable
        """
        self.environment_variables[name] = value
        self.log(f"Added environment variable: {name}")

    def get_tools_by_names(self, tool_names: List[Union[str, Dict[str, str]]] = None) -> Dict[str, Tool]:
        """
        Get specific tools by their names.

        Args:
            tool_names: List of tool names or dictionaries with name and version

        Returns:
            Dictionary of requested tools
        """
        self.ensure_initialized()

        if not tool_names:
            # If no specific tools requested, return all tools
            return {k: v for k, v in self.tools_map.items()}

        # If we're using local mode, we need to handle the tool loading differently
        if self.local_mode:
            return self.get_local_tools_by_names(tool_names)
        else:
            return self.get_api_tools_by_names(tool_names)

    def get_local_tools_by_names(self, tool_names: List[Union[str, Dict[str, str]]]) -> Dict[str, Tool]:
        """
        Get specific tools from local folder by their names without loading script content.
        """
        try:
            self.log(
                f"Loading {len(tool_names)} specific tools from folder...")

            # Load tools.json file
            tools_file_path = os.path.join(self.folder_path, "tools.json")
            if not os.path.exists(tools_file_path):
                raise FileNotFoundError(
                    f"tools.json file not found in {self.folder_path}")

            with open(tools_file_path, 'r') as f:
                tools_data = json.load(f)

            result = {}

            # Process each requested tool
            for name_param in tool_names:
                tool_name = name_param if isinstance(
                    name_param, str) else name_param["name"]
                version_name = None if isinstance(
                    name_param, str) else name_param.get("version")

                # Find the tool in the tools data
                tool_data = None
                for t in tools_data:
                    if t["name"].lower() == tool_name.lower():
                        tool_data = t
                        break

                if not tool_data:
                    self.log_error(f"Tool not found: {tool_name}")
                    continue

                # Get the versions dictionary
                versions = tool_data.get("versions", {})

                # Check if versions is a dictionary
                if not isinstance(versions, dict):
                    self.log_error(
                        f"Versions for tool {tool_name} is not a dictionary")
                    continue

                # If version is specified, use that version, otherwise use production version
                if version_name:
                    if version_name not in versions:
                        self.log_error(
                            f"Version {version_name} not found for tool {tool_name}")
                        continue
                    target_version = versions[version_name]
                else:
                    # Use production version
                    production_version_name = tool_data.get(
                        "production_version_name", "")
                    if not production_version_name or production_version_name not in versions:
                        self.log_error(
                            f"Production version not found for tool {tool_name}")
                        continue

                    target_version = versions[production_version_name]
                    version_name = production_version_name

                # Create script path instead of loading content
                script_extension = ".py" if target_version.get(
                    "script_type") == "python" else ".sh"
                script_filename = f"{tool_name}-{version_name}{script_extension}"
                script_path = os.path.join(self.folder_path, script_filename)

                if not os.path.exists(script_path):
                    self.log_error(f"Script file not found: {script_path}")
                    continue

                # Create the tool object without loading script content
                result[tool_name.lower()] = Tool(
                    id=tool_data.get("id", ""),
                    name=tool_data["name"],
                    description=target_version.get("description", ""),
                    input_schema=target_version.get("input_schema", {}),
                    script="",  # Empty script content
                    script_type=target_version.get("script_type", "bash"),
                    version_name=version_name,
                    script_path=script_path  # Store the path instead
                )

                self.log(
                    f"Loaded tool reference: {tool_name} (version: {version_name})")

            return result

        except Exception as e:
            self.log_error(
                f"Failed to load specific tools from folder: {str(e)}")

            # Fall back to cached tools if loading fails
            self.log("Falling back to cached tools", True)
            result = {}
            for name_param in tool_names:
                tool_name = name_param if isinstance(
                    name_param, str) else name_param["name"]
                tool_name_lower = tool_name.lower()

                if tool_name_lower in self.tools_map:
                    result[tool_name_lower] = self.tools_map[tool_name_lower]
                    self.log(
                        f"Using cached tool: {self.tools_map[tool_name_lower].name}")

            return result

    def get_api_tools_by_names(self, tool_names: List[Union[str, Dict[str, str]]]) -> Dict[str, Tool]:
        """
        Get specific tools from API by their names.

        Args:
            tool_names: List of tool names or dictionaries with name and version

        Returns:
            Dictionary of requested tools
        """
        try:
            self.log(f"Fetching {len(tool_names)} specific tools from API...")

            # Format tool names for the API request
            tools_request = []
            for name_param in tool_names:
                if isinstance(name_param, str):
                    tools_request.append({"name": name_param})
                else:
                    tools_request.append({
                        "name": name_param["name"],
                        "version": name_param.get("version")
                    })

            # Call the API endpoint with all tool names in a single request
            response = requests.post(
                f"{self.api_url}/api/get-individual-tools",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key
                },
                json={"tools": tools_request}
            )

            if response.status_code != 200:
                raise Exception(
                    f"API request failed with status {response.status_code}")

            result = response.json()

            if not result.get("tools") or not isinstance(result["tools"], list):
                raise Exception("Invalid response format from API")

            # Convert array to map with name as key (case insensitive)
            tools_map = {}
            for tool_data in result["tools"]:
                version = tool_data.get("version", {})
                name = tool_data["name"].lower()
                tools_map[name] = Tool(
                    id=tool_data.get("id", ""),
                    name=tool_data["name"],
                    description=version.get("description", ""),
                    input_schema=version.get("input_schema", {}),
                    script=version.get("script", ""),
                    script_type=version.get("script_type", "bash"),
                    version_name=version.get("version_name", "")
                )
                self.log(
                    f"Tool fetched: {tool_data['name']} (version: {version.get('version_name', 'latest')})")

            return tools_map

        except Exception as e:
            self.log_error(f"Failed to fetch individual tools: {str(e)}")

            # Fall back to cached tools if API request fails
            self.log("Falling back to cached tools", True)
            result = {}
            for name_param in tool_names:
                if isinstance(name_param, str):
                    tool_name = name_param.lower()
                else:
                    tool_name = name_param["name"].lower()

                if tool_name in self.tools_map:
                    result[tool_name] = self.tools_map[tool_name]
                    self.log(
                        f"Using cached tool: {self.tools_map[tool_name].name}")

            return result

    def execute_python_tool(self, tool: Tool, args: Dict[str, Any]) -> ToolExecutionResult:
        """
        Execute Python tool using the script content or file path.
        """
        try:
            # Use script content if available, otherwise load from file
            script_content = tool.script
            if not script_content and tool.script_path:
                self.log(f"Loading Python script from: {tool.script_path}")
                with open(tool.script_path, 'r') as f:
                    script_content = f.read()

            exec_locals = {}
            
            args["openv"] = self.environment_variables
            exec_locals['input_json'] = args

            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(script_content, globals(), exec_locals)

            output = stdout_capture.getvalue().strip()
            self.log(f"Python tool output: {output}")

            return ToolExecutionResult(output=output)

        except Exception as e:
            error_log = stderr_capture.getvalue().strip(
            ) if 'stderr_capture' in locals() else ""
            self.log(f"Python tool error: {error_log}")
            return ToolExecutionResult(error=f"Error executing Python tool: {error_log}\n{str(e)}")

    def execute_bash_tool(self, tool: Tool, args: Dict[str, Any]) -> ToolExecutionResult:
        """
        Execute bash tool using the script content or file path.
        """
        try:
            # Check if we have a script path (local mode) or script content
            if tool.script_path and os.path.exists(tool.script_path):
                # Execute script file by passing it to bash
                self.log(f"Executing bash script from: {tool.script_path}")

                # Create process by passing the script file to bash
                process = subprocess.Popen(
                    ["bash", tool.script_path],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            else:
                # Execute script content via bash
                process = subprocess.Popen(
                    ["bash", "-c", tool.script],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

            # Send JSON input
            args["openv"] = self.environment_variables
            stdin_data = json.dumps(args).encode()

            stdout, stderr = process.communicate(stdin_data)

            if process.returncode != 0:
                return ToolExecutionResult(error=f"Bash tool execution failed: {stderr.decode().strip()}")

            return ToolExecutionResult(output=stdout.decode().strip())

        except Exception as e:
            return ToolExecutionResult(error=f"Error executing Bash tool: {str(e)}")

    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> ToolExecutionResult:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            args: Arguments for the tool

        Returns:
            Tool execution result
        """
        self.ensure_initialized()

        tool_name_lower = tool_name.lower()
        if tool_name_lower not in self.tools_map:
            return ToolExecutionResult(error=f"Tool not found: {tool_name}")

        tool = self.tools_map[tool_name_lower]

        # Increment call count and check auto-refresh
        self.check_auto_refresh()

        if self.verbose:
            self.log(f"Executing tool: {tool.name}")
            self.log(f"Tool inputs: {json.dumps(args)}")

        # Execute based on script type
        if tool.script_type == "python":
            # Direct execution for Python tools
            return self.execute_python_tool(tool, args)
        elif tool.script_type == "bash":
            # Subprocess for Bash tools
            return self.execute_bash_tool(tool, args)
        else:
            return ToolExecutionResult(error=f"Unsupported script type: {tool.script_type}")

    def get_environment_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the current environment status.

        Returns:
            Current environment check results
        """
        return {k: v.to_dict() for k, v in self.environment_checks.items()}

    def get_tool_call_status(self) -> Dict[str, int]:
        """
        Get information about tool call counts and auto-refresh settings.

        Returns:
            Tool call status information
        """
        return {
            "callCount": self.tool_call_count,
            "autoRefreshCount": self.auto_refresh_count,
            "nextRefreshIn": (
                max(0, self.auto_refresh_count - self.tool_call_count)
                if self.auto_refresh_count > 0 else -1
            )
        }

    def set_auto_refresh_count(self, count: int) -> None:
        """
        Set the auto-refresh count.

        Args:
            count: Number of tool calls before auto-refresh (0 to disable)
        """
        if count < 0:
            raise ValueError(
                "Auto-refresh count must be a non-negative number")

        self.auto_refresh_count = count
        self.log(
            "Tool auto-refresh disabled" if count == 0
            else f"Tool auto-refresh set to occur every {count} tool calls",
            True
        )

    def set_verbose(self, enabled: bool) -> None:
        """
        Toggle verbose logging.

        Args:
            enabled: Whether to enable verbose logging
        """
        self.verbose = enabled
        self.log(
            f"Verbose logging {'enabled' if enabled else 'disabled'}", True)
