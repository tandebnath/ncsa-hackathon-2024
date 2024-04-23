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
      
 Now, the task for you take the generated subprompts:{sub_prompts} and the workflow:{workflow} and  Iautomate everything using ONLY Langchain (look at import statements I gave you. Give me code to create the automated workflow, based on the suggested workflow and each subprompt has a function created it for it which calls a function called code generatormodule which takes in the input of the subprompt for example: Write a Python script to plot and save a histogram of energy densities per channel, separating data by labels 0 and 1. The whole subprompt needs to be given to the code generator module). 

The codegenerator module code looks like this: {code_generator_code}. Please incorporate this into this code. 
Do not use chatprompttemplate, use llm.invoke to give the prompt and generate the code. 


"""
import os
from langchain_openai import AzureChatOpenAI
import subprocess

# Initialize the AzureChatOpenAI instance
llm = AzureChatOpenAI(
    azure_deployment="gpt-4-128k",
    openai_api_version=os.getenv("AZURE_MODEL_VERSION"),
    temperature=0,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

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

def generate_code(subprompt):
    """
    Generate code using the llm for the given subprompt.
    """
    print("Generating code for the subprompt...")
    result = llm.invoke(subprompt + "just give the function and the imports. Do not give any code descriptions or comments.")
    code = result.content
    print(code)
    return code

def save_code_to_file(code, filename):
    """
    Save the provided code to a .py file.
    """
    print(f"Saving code to {filename}...")
    with open(filename, 'w') as file:
        file.write(code)

def main():
    subprompts = {
        "bandpass_filter": """
    Write a Python function using SciPy to apply a bandpass filter to a 1D NumPy array representing time series data. The function should take the array, the sampling rate (2048 Hz), and the low (25 Hz) and high (500 Hz) cutoff frequencies as inputs and return the filtered data.
    """,
    "stft_preprocess": """
    Write a Python function to compute the Short-Time Fourier Transform (STFT) of a 1D NumPy array using SciPy. The function should take the array and the sampling rate (2048 Hz) as inputs and return the STFT results suitable for input into a CNN.
    """,
    "neural_network_components": """
    Create a Python script using PyTorch to define neural network components including a SqueezeExciteBlock, ConvBNSiLU, and an InceptionModule. These components should be integrated into a Model class that can be used for deep learning on time series data.
    """,
    "data_loader": """
    Write a Python class using PyTorch that loads .npy files from a given directory, applies a bandpass filter, and sets up data loaders for training a CNN. The class should handle cases where files are not found and skip to the next file.
    """,
    "training_script": """
    Write a Python training script using PyTorch that includes a loop for loss calculation, validation, and early stopping. The script should handle errors and log progress during the training of a CNN on time series data.
    """,
    "run_training": """
    Write a Python script to run the training process of a CNN using PyTorch. The script should use the previously defined data loader and training script, and it should output the training progress and final model performance.
    """,
    "cross_correlation": """
    Write a Python function to compute the cross-correlation between all possible pairs of channels in a 3-channel time series data array. The function should return the time lags between these channel pairs for arrays labeled with 1.
    """,
    "phase_space_plot": """
    Write a Python script to generate parametric phase space plots of the time derivative of the signal versus the signal for each channel in a 3-channel time series data array. The script should animate the plots, save the animations to an output folder, and handle both label 0 and label 1 data separately.
    """,
    "energy_density": """
    Write a Python function to compute the energy density of gravitational waves from a 3-channel time series data array. The function should perform a Fourier transform, calculate e(f), integrate it from 25 Hz to 500 Hz, and output a 1 by 3 array of the real part of the result for each channel. It should repeat this for all .npy files and record the energy densities.
    """,
    "energy_density_histogram": """
    Write a Python script to plot a histogram of energy densities per channel for label 0 and label 1 data separately. The script should save the histogram into an output folder.
    """,
    "spectrogram_plot": """
    Write a Python script to plot spectrograms of 3-channel time series data arrays. The script should take the STFT of the filtered data, plot the spectrogram in log frequency of base 2, merge three channels as RGB color channels, and save the merged spectrogram into an output folder for all files.
    """
    }

    for prompt_name, subprompt in subprompts.items():
        print(f"Processing subprompt: {prompt_name}")
        code = generate_code(subprompt)
        install_imports(code)
        save_code_to_file(code, f"{prompt_name}.py")
        print(f"Code for {prompt_name} saved successfully.")

if __name__ == "__main__":
    main()
