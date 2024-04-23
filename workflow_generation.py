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


#calling variables from first file
subprompts = {
    "subprompt_0": "Write a Python script using scipy to apply a bandpass filter from 25Hz to 500Hz on a 3 by n shape 1D time series numpy array. The sampling rate is 2048 Hz.",
    "subprompt_1": "Write a Python function to perform Short-Time Fourier Transform (STFT) on a time series data with a sampling rate of 2048 Hz using scipy.signal.",
    "subprompt_2": "Create a PyTorch neural network architecture with a SqueezeExciteBlock, ConvBNSiLU, and an InceptionModule integrated into a Model class.",
    "subprompt_3": "Write a Python class using PyTorch and numpy to load .npy files, apply preprocessing, and set up data loaders for training a CNN.",
    "subprompt_4": "Write a PyTorch training script that includes a training loop, loss calculation, validation, early stopping, and progress logging.",
    "subprompt_5": "Write a Python script to compute the cross-correlation between all possible pairs of channels in a 3-channel time series data to find the time lag, focusing only on data with label 1.",
    "subprompt_6": "Write a Python script to generate an animation of a parametric phase space plot of the time derivative of the signal versus the signal for each channel in a 3-channel time series data, and save the animation to an output folder.",
    "subprompt_7": "Write a Python function to compute the energy density of a gravitational wave from a bandpass filtered time series signal using Fourier transform and integration.",
    "subprompt_8": "Write a Python script to plot a histogram of energy densities for each channel, separating data with labels 0 and 1, and save the histogram to an output folder.",
    "subprompt_9": "Write a Python script to plot a spectrogram from STFT data, merge three channels into an RGB image, and save the spectrogram to an output folder."
}
workflow = {
    "parallel_tasks": [
        "subprompt_0",  # Bandpass filtering can be done in parallel for each file
        "subprompt_1",  # STFT can be done in parallel after bandpass filtering
        "subprompt_5",  # Cross-correlation can be done in parallel for label 1 data
        "subprompt_6",  # Phase space plot animations can be done in parallel
        "subprompt_7",  # Energy density computation can be done in parallel
        "subprompt_8",  # Histogram plotting can be done after energy density computation
        "subprompt_9"   # Spectrogram plotting can be done in parallel after STFT
    ],
    "sequential_tasks": [
        "subprompt_2",  # Neural network architecture needs to be defined before training
        "subprompt_3",  # Data loading and preprocessing class needs to be ready before training
        "subprompt_4"   # Training script needs the model and data loader to be defined first
    ]
}

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

# Existing imports and function definitions...

async def main(input_folder, output_folder):
    print("runnningg")
    # Generate code for parallel tasks
    parallel_results = await handle_parallel_tasks(workflow['parallel_tasks'])
    print(parallel_results)
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

