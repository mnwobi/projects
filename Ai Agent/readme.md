## Build an AI Agent
Time for Boot.dev to cash in on the AI hype.

If you've ever used Cursor or Claude Code as an "agentic" AI editor, you'll understand what we're building in this project.

We're building a toy version of Claude Code using Google's free Gemini API! As long as you have an LLM at your disposal, its actually surprisingly simple to build a (somewhat) effective custom agent.

## What Does the Agent Do?

The program we're building is a CLI tool that:

1. Accepts a coding task (e.g., "strings aren't splitting in my app, pweeze fix 🥺👉🏽👈🏽")
2. Chooses from a set of predefined functions to work on the task, for example:
   
- Scan the files in a directory
- Read a file's contents
- Overwrite a file's contents
- Execute the python interpreter on a file

3.Repeats step 2 until the task is complete (or it fails miserably, which is possible)
