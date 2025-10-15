import os 
from google.genai import types 

max_char=10000
def get_file_contents(working_directory, filepath):
    
    abs_dir= os.path.abspath(working_directory)
    abs_file= os.path.abspath(os.path.join(working_directory,filepath))

    if not abs_file.startswith(abs_dir):
        return f'Error: "{filepath}" is not in working directory'
    
    if os.path.isfile(abs_file) == False :
        return f"Error: {filepath} isn't a file "
    
    try:
        with open(abs_file,"r") as f:
            file_context=f.read(max_char)
            if len(file_context)>=max_char:
                file_context += f' ...File "{filepath} Trancated to {max_char} characters'
        return file_context
    except Exception as e:
        return f"Error while reading file : {e}"
    


schema_get_files_content = types.FunctionDeclaration(
    name="get_file_contents",
    description="Get the contents of a given file as a string, constrained to the working directory.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "filepath": types.Schema(
                type=types.Type.STRING,
                description="The path to the file from the working directory",
            ),
        },
    ),
)