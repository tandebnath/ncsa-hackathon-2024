from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env', override=True)
import os 
from typing import Dict, TypedDict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import END, StateGraph
from langchain_core.pydantic_v1 import BaseModel, Field
import langsmith
from langsmith.schemas import Example, Run

### Parameter
# Max tries
max_iterations = 3
# Reflect
# flag = 'reflect'
flag = 'do not reflect'
llm = AzureChatOpenAI(
            azure_deployment="gpt-4-128k",
            openai_api_version=os.getenv("AZURE_MODEL_VERSION"),
            temperature=0,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )

code_gen_prompt = ChatPromptTemplate.from_messages(
    [("system","""You are a coding assistant with expertise in LCEL, LangChain expression language. \n 
    Here is a full set of LCEL documentation:  \n ------- \n  {context} \n ------- \n Answer the user 
    question based on the above provided documentation. Ensure any code you provide can be executed \n 
    with all required imports and variables defined. Structure your answer with a description of the code solution. \n
    Then list the imports. And finally list the functioning code block. Here is the user question:"""),
    ("placeholder", "{messages}")]
)

# LCEL docs
url = "https://python.langchain.com/docs/expression_language/"
loader = RecursiveUrlLoader(
    url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text
)
docs = loader.load()

# Sort the list based on the URLs and get the text
d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
d_reversed = list(reversed(d_sorted))
concatenated_content = "\n\n\n --- \n\n\n".join(
    [doc.page_content for doc in d_reversed]
)

client = langsmith.Client()


# Data model
class code(BaseModel):
    """Code output"""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")
    description = "Schema for code solutions to questions about LCEL."



code_gen_chain = code_gen_prompt | llm.with_structured_output(code)

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        error : Binary flag for control flow to indicate whether test error was tripped
        messages : With user question, error messages, reasoning
        generation : Code solution
        iterations : Number of tries 
    """

    error : str
    messages : List
    generation : str
    iterations : int
    

 
### Nodes

def generate(state: GraphState):
    """
    Generate a code solution

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation
    """

    print("---GENERATING CODE SOLUTION---")
    
    # State
    messages = state["messages"]
    iterations = state["iterations"]
    error = state["error"]

    # We have been routed back to generation with an error
    if error == "yes":
        messages += [("user","Now, try again. Invoke the code tool to structure the output with a prefix, imports, and code block:")]
        
    # Solution
    code_solution = code_gen_chain.invoke({"context": concatenated_content, "messages" : messages})
    messages += [("assistant",f"{code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}")]
    
    # Increment
    iterations = iterations + 1
    return {"generation": code_solution, "messages": messages, "iterations": iterations}

def code_check(state: GraphState):
    """
    Check code

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, error
    """

    print("---CHECKING CODE---")
    
    # State
    messages = state["messages"]
    code_solution = state["generation"]
    iterations = state["iterations"]

    # Get solution components
    prefix = code_solution.prefix
    imports = code_solution.imports
    code = code_solution.code

    # Check imports
    try:
        exec(imports)
    except Exception as e:
        print("---CODE IMPORT CHECK: FAILED---")
        error_message = [("user", f"Your solution failed the import test: {e}")]
        messages += error_message
        return {"generation": code_solution, "messages": messages, "iterations": iterations, "error": "yes"}
    
    # Check execution
    try:
        exec(imports + "\n" + code)
    except Exception as e:
        print("---CODE BLOCK CHECK: FAILED---")
        error_message = [("user", f"Your solution failed the code execution test: {e}")]
        messages += error_message
        return {"generation": code_solution, "messages": messages, "iterations": iterations, "error": "yes"}
  
    # No errors
    print("---NO CODE TEST FAILURES---")
    return {"generation": code_solution, "messages": messages, "iterations": iterations, "error": "no"}

def reflect(state: GraphState):
    """
    Reflect on errors

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation
    """

    print("---GENERATING CODE SOLUTION---")
    
    # State
    messages = state["messages"]
    iterations = state["iterations"]
    code_solution = state["generation"]

    # Prompt reflection
    reflection_message = [("user", """You tried to solve this problem and failed a unit test. Reflect on this failure
                                    given the provided documentation. Write a few key suggestions based on the 
                                    documentation to avoid making this mistake again.""")]
    
    # Add reflection
    reflections = code_gen_chain.invoke({"context"  : concatenated_content, "messages" : messages})
    messages += [("assistant" , f"Here are reflections on the error: {reflections}")]
    return {"generation": code_solution, "messages": messages, "iterations": iterations}

### Edges

def decide_to_finish(state: GraphState):
    """
    Determines whether to finish.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    error = state["error"]
    iterations = state["iterations"]

    if error == "no" or iterations == max_iterations:
        print("---DECISION: FINISH---")
        return "end"
    else:
        print("---DECISION: RE-TRY SOLUTION---")
        if flag == 'reflect':
            return "reflect"
        else:
            return "generate"


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("generate", generate)  # generation solution
workflow.add_node("check_code", code_check)  # check code
workflow.add_node("reflect", reflect)  # reflect

# Build graph
workflow.set_entry_point("generate")
workflow.add_edge("generate", "check_code")
workflow.add_conditional_edges(
    "check_code",
    decide_to_finish,
    {
        "end": END,
        "reflect": "reflect",
        "generate": "generate",
    },
)
workflow.add_edge("reflect", "generate")
app = workflow.compile()
question = "How can I directly pass a string to a runnable and use it to construct the input needed for my prompt?"
app.invoke({"messages":[("user",question)],"iterations":0})

def check_import(run: Run, example: Example) -> dict: 
    imports = run.outputs.get("imports")
    try:
        exec(imports)
        return {"key": "import_check" , "score": 1} 
    except:
        return {"key": "import_check" , "score": 0} 

def check_execution(run: Run, example: Example) -> dict: 
    imports = run.outputs.get("imports")
    code = run.outputs.get("code")
    try:
        exec(imports + "\n" + code)
        return {"key": "code_execution_check" , "score": 1} 
    except:
        return {"key": "code_execution_check" , "score": 0} 

def predict_langgraph(example: dict):
    """ LangGraph """
    graph = app.invoke({"messages":[("user",example["question"])],"iterations":0})
    solution = graph["generation"]
    return {"imports": solution.imports, "code": solution.code}


# Evaluator
code_evalulator = [check_import,check_execution]

user_input = {
    "question": "Please write python code to read a medical image and do denoising on it. Use cv2 if needed. The entire code will provide a function interface in the end. The input is image_path, output is a numpy array(H, W, 3) representing the denoised image"
}

result = predict_langgraph(user_input)
print(result)

# write the code to a file
with open("temp.py", "w") as file:
    file.write(result["imports"])
    file.write("\n")
    file.write(result["code"])