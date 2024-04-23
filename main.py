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
    #give the user prompt to the llm with the system template
    user_prompt=prompt
    systemtemplate = f"""
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
        Now, the task for you take the user input:{user_prompt} 
        I don not want code for any step. I want PROMPTS for every step, that I can feed to AzureChatOpenAI, and get code for.
        Now I'm not going to feed prompts either. I want you to automate everything using ONLY Langchain (look at import statements I gave you)
        Give me code to create the automated workflow, and then tell me what prompts to add where to make it go from input to output. Generate the prompts for each step based on the user inputs and then give it to the llm invoke later. make sure there is code taking input folder and then outputting to an output folder. Do not use chatprompttemplate, use llm.invoke to give the prompt and generate the code. Make sure that the subprompts are generated for the user inputs. """
    print(systemtemplate)
    with tqdm(total=100, desc="Processing", unit="step") as progress:
        progress.update(10)  # Example update, adjust as needed
        llmresponse=llm.invoke(systemtemplate)
        llm_response_context=llmresponse.content
        progress.update(10)  # Example update, adjust as needed
    #writing the output into another file
    with open("main_script.py", "w") as file:
        file.write(llm_response_context)
    return llm_response_context

@app.command()
def run(prompt_file: str = typer.Argument(..., help="Path to the file containing the prompt.")):
    """Gravitational Wave Detection Tool."""
    with open(prompt_file, "r") as file:
        prompt = file.read()
    typer.echo("--------------------------------------------------")
    typer.echo("          Gravitational Waves Detection Tool")
    typer.echo("         By Team 1 - Team InitToWinIt")
    typer.echo("--------------------------------------------------")
    typer.echo("Generating Main Script Code from your Prompt")
    result = call_azure_openai(prompt)
    typer.echo("Response from Azure OpenAI:")
    typer.echo(result)

if __name__ == "__main__":
    app()