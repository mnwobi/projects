# Google Gemini AI Coding Agent

An intelligent AI-powered coding assistant built with Google's Gemini AI that can perform various file operations, code execution, and project management tasks through natural language commands.

## 🤖 Overview

This project implements an AI coding agent using Google's Gemini 2.0 Flash model that can understand natural language requests and execute various coding-related tasks. The agent can read and write files, run Python code, analyze project structures, and assist with software development workflows.

## ✨ Key Features

- **🔍 File System Operations** - List directories, read file contents, and navigate project structures
- **📝 Code Execution** - Run Python files with custom arguments and capture outputs
- **✏️ File Management** - Create, edit, and overwrite files with AI-generated content
- **🧠 Intelligent Planning** - AI analyzes requests and creates execution plans before taking action
- **🔒 Security Focused** - All operations are constrained to the working directory for safety
- **⚡ Real-time Responses** - Fast execution with Google's Gemini 2.0 Flash model

## 🏗️ Architecture

### Core Components

```
Google Gemini AI Coding Agent/
├── main.py                 # Main agent entry point
├── .env                    # API key configuration
├── Functions/              # Core function implementations
│   ├── get_files_info.py   # Directory listing functionality
│   ├── get_file_content.py # File reading operations
│   ├── write_file.py       # File creation and editing
│   ├── run_python_file.py  # Python code execution
│   └── call_function.py    # Function call orchestration
└── calculator/             # Example project for testing
```

### Function Capabilities

| Function | Description | Use Case |
|----------|-------------|----------|
| `get_files_infos` | Lists files and directories with metadata | Project exploration and navigation |
| `get_file_contents` | Reads and returns file contents | Code analysis and documentation |
| `write_files` | Creates or overwrites files | Code generation and file creation |
| `run_python_file` | Executes Python files with arguments | Testing and running applications |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google Gemini API key
- Required Python packages: `google-genai`, `python-dotenv`

### Installation



2. **Install dependencies**
   ```bash
   pip install google-genai python-dotenv
   ```

3. **Configure API key**
   ```bash
   # Create .env file with your Gemini API key
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   ```

4. **Run the agent**
   ```bash
   python main.py "your coding request here"
   ```

## 💻 Usage Examples

### Basic Commands

```bash
# List files in current directory
python main.py "list all files in the current directory"

# Read a specific file
python main.py "read the contents of main.py"

# Create a new Python file
python main.py "create a hello world script called hello.py"

# Run a Python file
python main.py "run the calculator script with arguments 10 + 5"

# Analyze project structure
python main.py "analyze the project structure and explain what each component does"
```

### Advanced Usage

```bash
# Complex multi-step tasks
python main.py "create a Flask web application with routes for home, about, and contact pages"

# Code refactoring
python main.py "refactor the existing code to use more efficient algorithms"

# Documentation generation
python main.py "generate comprehensive documentation for all functions in the project"
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Your Google Gemini API key | Yes |

### Command Line Options

- **Verbose mode**: Add `--verbose` flag for detailed execution logs
- **Custom prompts**: Pass any coding request as a command line argument

## 🛠️ Development


### Function Template

```python
from google.genai import types

def my_new_function(parameter1, parameter2):
    # Implementation here
    return result

schema_my_new_function = types.FunctionDeclaration(
    name="my_new_function",
    description="Description of what this function does",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "parameter1": types.Schema(type=types.Type.STRING),
            "parameter2": types.Schema(type=types.Type.INTEGER),
        },
    ),
)
```

## 🎯 Use Cases

### For Developers
- **Code Generation** - Generate boilerplate code, functions, and classes
- **Project Setup** - Create new projects with proper structure
- **Debugging Assistance** - Analyze code and suggest fixes
- **Documentation** - Generate README files and code comments
- **Refactoring** - Improve existing code structure and efficiency

### For Students
- **Learning Assistant** - Get help with programming assignments
- **Code Explanation** - Understand complex algorithms and patterns
- **Project Templates** - Start new projects with best practices

### For Teams
- **Code Review** - Automated analysis and suggestions
- **Documentation Updates** - Keep project docs current
- **Onboarding** - Help new developers understand codebase

## 🔒 Security & Limitations

- **Working Directory Constraint** - All file operations are limited to the project directory
- **Function-based Architecture** - Only predefined functions can be executed
- **API Rate Limits** - Subject to Google Gemini API limitations
- **No External Network Access** - Cannot make HTTP requests or access external resources

## 🐛 Troubleshooting

### Common Issues

1. **API Key Issues**
   - Ensure `GEMINI_API_KEY` is correctly set in `.env`
   - Check that the API key has sufficient permissions

2. **Function Execution Errors**
   - Verify all required parameters are provided
   - Check file paths are relative to working directory

3. **Import Errors**
   - Install missing dependencies: `pip install google-genai python-dotenv`

## 📈 Performance Tips

- Use specific, clear prompts for better results
- Break complex tasks into smaller, manageable steps
- Enable verbose mode (`--verbose`) for debugging
- Keep requests focused on coding tasks for optimal performance

## 🤝 Contributing

This is a personal project exploring AI-assisted software development. Contributions, suggestions, and feedback are welcome!

## 📄 License

MIT License

## 🔗 Links

- [Google Gemini API Documentation](https://ai.google.dev/)
- [Python Client Library](https://github.com/google/generative-ai-python)

---

*Built with Google's Gemini AI for intelligent coding assistance*
