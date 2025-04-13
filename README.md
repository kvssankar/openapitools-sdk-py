# OpenAPITools SDK

## Introduction

OpenAPITools Python package enables developers to manage, and execute tools across multiple AI API providers. It provides a unified interface for working with tools in Anthropic's Claude, OpenAI's GPT models, and LangChain frameworks.

With OpenAPITools, you can:

- Create tools as Python or Bash scripts with standardized input/output
- Access these tools through a single, consistent SDK
- Integrate tools with Claude, GPT, and LangChain models
- Build interactive chatbots that can use tools to solve complex tasks

## Installation

### Prerequisites

- Python 3.8 or later
- Access keys to at least one of the supported AI providers (Anthropic, OpenAI, or LangChain)
- Get an API key for [OpenAPITools](https://openapitools.com) from the [Settings](https://openapitools.com/settings) page

### Install from PyPI

```bash
pip install reacter-openapitools requests
```

If you're using the **LangChain adapter**, you’ll also need to install `langchain` and `langchain-core`:

```bash
pip install langchain langchain-core
```


## Tool Execution Details

### Python Tools

- Python tools are executed using Python's `exec()` function directly in the current process
- Benefits:
  - No interpreter startup overhead
  - Full privacy (code runs locally)
  - Faster execution compared to subprocess methods
- Python tools receive arguments via an `input_json` dictionary and can access environment variables through `input_json["openv"]`

### Bash Tools

- Bash tools are executed as subprocesses
- Arguments are passed as JSON to the script's standard input
- Recommended for non-Python environments for better performance
- Note: Bash tools should be tested in Linux environments or WSL, as they may not function correctly in Windows

## Usage Modes

### [Local Mode](/localmode) (preferred)

```python
adapter = ToolsAdapter(folder_path="/path/to/tools")
```

### API Mode (rate limits apply)

```python
adapter = ToolsAdapter(api_key="your_api_key")
```

## Performance Considerations

- **Python Tools**: Best for Python environments, executed in-process with minimal overhead
- **Bash Tools**: Better for non-Python servers or when isolation is needed
- For maximum performance in non-Python environments, prefer Bash tools

## Security and Privacy

- All tool execution happens locally within your environment
- No code is sent to external servers for execution
- Environment variables can be securely passed to tools

## Integration with AI Models

OpenAPITools provides native integration with:

- Anthropic's Claude
- OpenAI's GPT models
- LangChain frameworks

This allows you to build AI assistants that can leverage tools to perform complex tasks.

Visit [docs.openapitools.com](https://docs.openapitools.com) for more information on how to use the OpenAPITools SDK, including detailed examples and API references.