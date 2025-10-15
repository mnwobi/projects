from google import genai
from google.genai import types
from dotenv import load_dotenv
import os 
import sys

from Functions.get_files_info import get_files_infos, schema_get_files_info
from Functions.get_file_content import get_file_contents,schema_get_files_content
from Functions.write_file import write_files,schema_write_files
from Functions.run_python_file import run_python,schema_run_python_file

from Functions.call_function import call_function



# loading in model
load_dotenv()
api_key = os.environ.get('GEMINI_API_KEY')

client=genai.Client(api_key=api_key)


def main ():
    # gettign prompt
    verbose_flag=False

    if len(sys.argv)<2:
        print('I need a statement as an argument')
        return

    if len(sys.argv)==3 and len(sys.arg[2]=='--verbose'):
        verbose_flag=True 

    prompt= sys.argv[1]
    #response 

    system_prompt = """
You are a helpful AI coding agent.

When a user asks a question or makes a request, make a function call plan. You can perform the following operations:

- List files and directories
- Read file contents
- Execute Python files with optional arguments
- Write or overwrite files 

When the user asks about the code project - they are referring to the working directory.
So, you should typically start by looking at the project's files, and figuring out how to run the 
project and how to run its tests, you'll always want to test the tests and the actual project
 to verify that behavior is working.

All paths you provide should be relative to the working directory. You do not need to specify the working directory in your function calls as it is automatically injected for security reasons.
"""
    # users message to LLM
    messages=[ 
        types.Content(role='user', parts=[types.Part(text=prompt)]),
    ]

    available_functions = types.Tool(
    function_declarations=[
        schema_get_files_info,schema_get_files_content,schema_write_files,schema_run_python_file
    ]
)   
    config=types.GenerateContentConfig(
    tools=[available_functions], system_instruction=system_prompt
)
    
    # Building agentinc part of AI

    max_iter= 20 # number of max trys/loops 
    for i in range(0,max_iter):
         
        response= client.models.generate_content(model= 'gemini-2.0-flash-001', contents=messages,
                                                config=config)
        
        # if functions is called then this will be a tool type of messgage that will go to the llm

        if response.candidates:
             for candidate in response.candidates:
                if candidate is None or response.candidates is None:
                    continue 
                messages.append(candidate.content) 

        if response.function_calls:
            for function_call in response.function_calls:
                result= call_function(function_call)
                messages.append(result)
                # print(result)

        else:
            # final agent text messgae
            return print(response.text)
        
        if response is None or response.usage_metadata is None:
                return print( 'response is maleformed')
        


    
main()

# print( run_python("calculator","main.py",['9 + 22']))