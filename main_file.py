import typer
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
from tqdm import tqdm
import subprocess

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
    Now, the task for you take the user input:{prompt} I want you to generate the sub prompts for every step, that I can feed to AzureChatOpenAI, and get code for. The final output i expect from you is a dictionary called subprompts and all the subprompts should be stored in it. Also in the output, i want you to determine which subprompts can be run sequentially and which can be run parallely in a workflow and create a dictionary called workflow which has two keys: parallel_tasks and sequential_tasks contains the names of the subprompts. Make sure the subprompts are of good quality for llm understanding and code generation.
    Make sure that the subprompts are generated for the user inputs. """
    # Send the initial prompt to the AzureChatOpenAI
    initial_response = llm.invoke(initial_prompt)
    with open("initial_script.py", "w") as file:
        file.write(initial_response.content)
    # Feedback check prompt to verify completeness and correctness
    feedback_prompt = f"""check whether each subprompt has been generated in the {initial_response.content}. if anysubprompt is MISSING, please add it in the script based on the user prompt. From you, now i expect the full code with the changes you made if you added any missing subprompts. Make sure that you did not MISS OUT on any subprompts. I want you to output only the subprompts dictionary and the workflow dictionary """
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
    typer.echo("Running workflow_generation.py...")
    subprocess.run(["python", "workflow_generation.py", str(result.content)])
    typer.echo("workflow_generation.py executed.")

if __name__ == "__main__":
    app()
