import os 
from google.genai import types 

def write_files(working_dir, file_path,content):

    abs_dir= os.path.abspath(working_dir)
    abs_file= os.path.abspath(os.path.join(working_dir,file_path))

    if not abs_file.startswith(abs_dir):
        return f'Error: "{file_path}" is not in working directory'
    
    parent_dir=os.path.dirname(abs_file)

    if not os.path.isdir(parent_dir): 
        try:
            os.makedirs(parent_dir )
        except Exception as e :
            return f"Couldn't create parent dir: {parent_dir} = {e}"
    
    # creating new files and overwritting exisiting 
    try:
        with open(abs_file,'w') as f:
            f.write(content)
        return f"I successfully wrote {abs_file} with content length {len(content)}"
    except Exception as e:
        return f"Failed to write to {abs_file}: {e}"


schema_write_files = types.FunctionDeclaration(
    name="write_files",
    description="Overwrites or writes to an existing python file and can also have the ability to create a new python file if it doesn;t exist(and creates parent dirs safely). Constrained to the working directory.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "file_path": types.Schema(
                type=types.Type.STRING,
                description="The file to write relative to the working directory",
            ),
            "content": types.Schema(
                type=types.Type.STRING,
                description="The content which is a string  that should be added to the file",
            )
        },
    ),
)