import os
import subprocess
from google.genai import types 


def run_python(working_dir, file_path, args=[]):

    abs_dir = os.path.abspath(working_dir)
    abs_file = os.path.abspath(os.path.join(working_dir, file_path))

    if not abs_file.startswith(abs_dir):
        return f'Error: "{file_path}" is not in working directory'
    if not os.path.isfile(abs_file):
        return f'Error: "{file_path}" is not a file'
    if not file_path.endswith('.py'):
        return f'Error: "{file_path}" is not a python file'
    
    try:
        final_arg=['python3',file_path]
        final_arg.extend(args)
        output = subprocess.run(
            final_arg,
            cwd=abs_dir, 
            timeout=30, 
            capture_output=True,
            text=True  # Correctly decodes the output to strings
        )

        # Build the final output string
        final_string = f"STDOUT:\n{output.stdout}\nSTDERR:\n{output.stderr}"

        # Handle successful execution
        if output.returncode == 0:
            return final_string
        
        # Handle non-zero exit code
        else:
            final_string += f"\nProcess exited with code {output.returncode}"
            return final_string

    except subprocess.TimeoutExpired:
        return f"Error: Python file '{file_path}' timed out after 30 seconds."
    except Exception as e:
        return f"Error executing python file '{file_path}': {e}"


schema_run_python_file = types.FunctionDeclaration(
    name="run_python",
    description="The ability to run a python/.py file with the python 3 as an interperter. Accepts additional CLI Args as an optional array ",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "file_path": types.Schema(
                type=types.Type.STRING,
                description="The file to run relative to the working directory, ",
            ),
            # an array of string so mkae it nested
            "args": types.Schema(
                type=types.Type.ARRAY,
                description="An optional array of strings of CLI args to be used for the python file.",
                items=types.Schema(
                    type=types.Type.STRING,
                )
            )
        },
    ),
)