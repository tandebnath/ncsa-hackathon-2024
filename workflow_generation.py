"""
Prompt:You are a highly skilled AI assistant proficient in various domains including software development, genAI, LLMs, and full-stack technologies. Your task is to write a main script based on a user prompt provided as input.
    Now please use these import statements in the code that I ask you to generate.
    from langchain_openai import AzureChatOpenAI, ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.pydantic_v1 import BaseModel, Field
    from langgraph.graph import END, StateGraph
    from langchain_core.pydantic_v1 import BaseModel, Field
    import langsmith
    import subprocess
    from langsmith.schemas import Example, Run
    This is how the azurechatopenai works:
    llm = AzureChatOpenAI(
            azure_deployment="gpt-4-hackathon",
            openai_api_version=os.getenv("AZURE_MODEL_VERSION"),
            temperature=0,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )
      
 Now, the task for you take the generated subprompts:{sub_prompts} and the workflow:{workflow} and  automate everything using ONLY Langchain (look at import statements I gave you. Give me code that takes the subprompts and workflow and generates the whole end to end code. 

Keep in mind that the workflow needs to run end to end, if you generate code for each subprompt, the result of that code might be used for the next subprompt, so be context aware in that matter.

This code takes in an input folder and gives the output in an output folder.

Do not use chatprompttemplate, use llm.invoke to give the prompt and generate the code. 


"""
import os
from langchain_openai import AzureChatOpenAI
import subprocess
import asyncio
import main_file

def install_imports(code):
    """
    Install the required imports for the generated code.
    """
    print("Installing required imports...")
    requirements = set()
    for line in code.splitlines():
        if line.startswith('import') or line.startswith('from'):
            try:
                # Attempt to import the module to check if it's already installed
                __import__(line.split()[1])
            except ImportError as e:
                # Module not installed, add it to the requirements
                requirements.add(line.split()[1])

    # AzureChatOpenAI instance
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4-hackathon",
        openai_api_version=os.getenv("AZURE_MODEL_VERSION"),
        temperature=0,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )

    for req in requirements:
        # Generate pip install command using Azure OpenAI based on the missing import error
        pip_install_prompt = f"I want your answer to return ONLY a pip install statement for the missing library: {req}"
        response = llm.invoke(pip_install_prompt)
        content = response.content.strip("`").replace("```plaintext\n", "").replace("```", "")

        try:
            # Execute the generated pip install command
            subprocess.run(content, shell=True, check=True)
            print(f"Library installation successful for {req}.")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while installing the library {req}: {e}")
            return {"error": f"Your solution failed the import test: {e}"}

    print("All required libraries installed successfully.")
    return {"error": "no"}

# Initialize the AzureChatOpenAI with the necessary environment variables
llm = AzureChatOpenAI(
    azure_deployment="gpt-4-hackathon",
    openai_api_version=os.getenv("AZURE_MODEL_VERSION"),
    temperature=0,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

# Define a function to asynchronously invoke the LLM to generate code for a given prompt
async def generate_code(prompt):
    response = await llm.invoke(prompt)
    install_imports(response.content)
    print(response.content)
    return response.content

import sys




input_folder = 'input'
output_folder = 'output'

# Define a function to handle the parallel execution of tasks
async def handle_parallel_tasks(tasks):
    coroutines = [generate_code(str(subprompts[task])+"just give the function and the imports. Do not give any code descriptions or comments.") for task in tasks]
    print(coroutines)
    results = await asyncio.gather(*coroutines)
    return results

# Define a function to handle the sequential execution of tasks
async def handle_sequential_tasks(tasks):
    results = []
    for task in tasks:
        code = await generate_code(str(subprompts[task])+"just give the function and the imports. Do not give any code descriptions or comments.")
        results.append(code)
    return results

import re
import ast
main_response=sys.argv[1] 
print("the main response")
print(main_response)

subprompts_str = re.search(r"subprompts\s*=\s*{[^}]+}", main_response, re.MULTILINE).group(0)
print(subprompts_str)
extracted_string = subprompts_str.strip()
dictionary_string = extracted_string.split('=', 1)[1]
subprompts = ast.literal_eval(dictionary_string)


workflow_str = re.search(r"workflow\s*=\s*{[^}]+}", main_response, re.MULTILINE).group(0)
extracted_string = workflow_str.strip()
dictionary_string = extracted_string.split('=', 1)[1]
workflow= ast.literal_eval(dictionary_string)



async def main(input_folder, output_folder):
    main_response=sys.argv[1] 
    main_response=main_response.replace("```python", "").strip()
    # Generate code for parallel tasks
    parallel_results = await handle_parallel_tasks(workflow['parallel_tasks'])
    # Save the generated code for parallel tasks to the output folder
    for i, result in enumerate(parallel_results):
        task_name = workflow['parallel_tasks'][i]
        with open(os.path.join(output_folder, f"{task_name}.py"), 'w') as f:
            f.write(result)
    
    # Generate and execute code for sequential tasks
    sequential_results = await handle_sequential_tasks(workflow['sequential_tasks'])
    print(sequential_results)
    # Save and execute the generated code for sequential tasks
    for i, result in enumerate(sequential_results):
        task_name = workflow['sequential_tasks'][i]
        file_path = os.path.join(output_folder, f"{task_name}.py")
        with open(file_path, 'w') as f:
            f.write(result)
        # Execute the script (assuming that each script is self-contained and executable)
        subprocess.run(["python", file_path], check=True, cwd=input_folder)  # Specify the input folder

# Run the main function with the specified input and output folders
print("hello")
asyncio.run(main(input_folder, output_folder))

