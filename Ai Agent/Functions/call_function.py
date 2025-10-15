from Functions.get_files_info import get_files_infos, schema_get_files_info
from Functions.get_file_content import get_file_contents,schema_get_files_content
from Functions.write_file import write_files,schema_write_files
from Functions.run_python_file import run_python,schema_run_python_file

from google.genai import types 





def call_function(function_call_part, verbose=False):
    if verbose:
        print(f"Calling function: {function_call_part.name}({function_call_part.args})")
    else: 
        print(f" - Calling function: {function_call_part.name}")

    working_dir='calculator'

    # actually calling functions 

    result=""

    if function_call_part.name == "get_file_contents":
       result=get_file_contents(working_dir ,**function_call_part.args)
    
    if function_call_part.name == "write_files":
       result=write_files(working_dir ,**function_call_part.args)

    if function_call_part.name == "get_files_infos":
       result= get_files_infos(working_dir ,**function_call_part.args)

    if function_call_part.name == "run_python":
        result=run_python(working_dir ,**function_call_part.args)

    # return an error message
    if result =="":
        return types.Content(
    role="tool",
    parts=[
        types.Part.from_function_response(
            name=function_call_part.name,
            response={"error": f"Unknown function: {function_call_part.name}"},
        )
    ],
)
    # return results 
    else: 
        return types.Content(
    role="tool",
    parts=[
        types.Part.from_function_response(
            name=function_call_part.name,
            response={"result": result},
        )
    ],
)