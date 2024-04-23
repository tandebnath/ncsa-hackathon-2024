import typer
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(dotenv_path='.env', override=True)

app = typer.Typer()

def call_azure_openai(prompt: str) -> dict:
    """Function to call Azure OpenAI API."""
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4-hackathon",
        openai_api_version=os.getenv("AZURE_MODEL_VERSION"),
        temperature=0,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )

    # Initial prompt processing with the user's input
    initial_prompt = f"""
    You are a highly skilled AI assistant proficient in various domains including software development, genAI, LLMs, and full-stack technologies. Your task is to write a main script based on a user prompt provided as input.
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
        and you need to invoke llm.invoke when you need to make any call to the azurechatopenai.
        Now, the task for you take the user input:{prompt} 
        I don not want code for any step. I want PROMPTS for every step, that I can feed to AzureChatOpenAI, and get code for.
        Now I'm not going to feed prompts either. I want you to automate everything using ONLY Langchain (look at import statements I gave you)
        Give me code to create the automated workflow, and then tell me what prompts to add where to make it go from input to output. Generate the prompts for each step based on the user inputs and then give it to the llm invoke later. make sure there is code taking input folder and then outputting to an output folder. Do not use chatprompttemplate, use llm.invoke to give the prompt and generate the code. Make sure that the subprompts are generated for the user inputs. """
    # Send the initial prompt to the AzureChatOpenAI
    initial_response = llm.invoke(initial_prompt)
    with open("initial_script.py", "w") as file:
        file.write(initial_response.content)
    # Feedback check prompt to verify completeness and correctness
    feedback_prompt = f"""check whether each subprompt has been generated. if anysubprompt is MISSING, please add it in the script based on the user prompt. From you, now i expect the full code with the changes you made if you added any missing subprompts. Make sure that you did not MISS OUT on any subprompts. I want the code to be immeditately runnable, do not tell me to add any logic, all the logic and prompt should be added by you and ensure that all the subprompts are defined in your final resultant code that you generate. Check and verify if the generated code:{initial_response} covers all required aspects from the user prompt:{initial_prompt}. """
    # Send the feedback prompt for a mandatory check
    final_response = llm.invoke(feedback_prompt)
    with open("feedback_script.py", "w") as file:
        file.write(final_response.content)
    return final_response

@app.command()
def run(prompt_file: str = typer.Argument(..., help="Path to the file containing the prompt.")):
    """Gravitational Wave Detection Tool."""
    with open(prompt_file, "r") as file:
        user_prompt = file.read()

    typer.echo("--------------------------------------------------")
    typer.echo("          Gravitational Waves Detection Tool       ")
    typer.echo("             By Team 1 - Team InitToWinIt          ")
    typer.echo("--------------------------------------------------")
    typer.echo("Generating Main Script Code from your Prompt")
    result = call_azure_openai(user_prompt)
    typer.echo("Final Response from Azure OpenAI after Verification:")
    typer.echo(result)

if __name__ == "__main__":
    app()
