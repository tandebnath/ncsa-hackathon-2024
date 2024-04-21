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
max_unit_iterations = 5
# Reflect
# flag = 'reflect'
unit_flag = True
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

# # LCEL docs
# url = "https://python.langchain.com/docs/expression_language/"
# loader = RecursiveUrlLoader(
#     url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text
# )
# docs = loader.load()

# # Sort the list based on the URLs and get the text
# d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
# d_reversed = list(reversed(d_sorted))
# concatenated_content = "\n\n\n --- \n\n\n".join(
#     [doc.page_content for doc in d_reversed]
# )

client = langsmith.Client()

# Data model
class code(BaseModel):
    """Code output"""
    name: str = Field(description="Name of the code block")
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")
    description = "Schema for code solutions to questions about LCEL."

class CodeBlock:
    def __init__(self) -> None:
        self.prefix = ""
        self.imports = ""
        self.code = ""
        self.unit_test_code = ""
        
class CodeBlocks:
    def __init__(self) -> None:
        self.blocks = [CodeBlock()]
        self.user_inputs = []
        
    def set_latest_block(self, attribute, value):
        setattr(self.blocks[-1], attribute, value)
    
    def finish_block(self):
        self.blocks.append(CodeBlock())
    
    def return_excutable_block(self, idx, unit_test=True):
        block = self.blocks[idx]
        if unit_test:
            return "\n".join([ block.imports + "\n" + block.code + "\n" + block.unit_test_code])
        else:
            return "\n".join([ block.imports + "\n" + block.code])
        # return "\n".join([ block.imports + "\n" + block.code + "\n" + block.unit_test_code])
    
    def get_latest_user_input(self):
        return self.user_inputs[len(self.blocks) - 1]
    
    def save_all_files(self, dir_name="tempcode"):
        os.makedirs(dir_name, exist_ok=True)
        for i, block in enumerate(self.blocks[:-1]):
            with open(f"{dir_name}/{block.name}.py", "w") as file:
                file.write(self.return_excutable_block(i))
    

code_gen_chain = code_gen_prompt | llm.with_structured_output(code)
code_gen_result = CodeBlocks()

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
    code_solution = code_gen_chain.invoke({"context":"","messages" : messages})
    messages += [("assistant",f"{code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}")]
    
    code_gen_result.set_latest_block("prefix", code_solution.prefix)
    code_gen_result.set_latest_block("imports", code_solution.imports)
    code_gen_result.set_latest_block("code", code_solution.code)
    code_gen_result.set_latest_block("name", code_solution.name)
    # Increment
    iterations = iterations + 1
    return {"generation": code_solution, "messages": messages, "iterations": iterations}

def generate_unit_tests(state: GraphState):
    """
    Generate unit tests

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation
    """

    print("---GENERATING UNIT TESTS---")
    
    # State
    messages = state["messages"]
    iterations = state["iterations"]

    # get user input
    # user_input = input("Please provide the input for the unit test in format (arg1, arg2, ...), give \n if no input:")
    user_input = code_gen_result.get_latest_user_input()
    if user_input != "\n":
        messages += [("user",f"input arguments for the unit test: {user_input}")]
    # Generate unit tests
    messages += [("user",f"Now, Please generate the test code for the the solution to see if the function can run: {code_gen_result.return_excutable_block(-1, True)}, you can generate the input by passing randomly built matrix. This code will be directly excecuted so not make it a function")]
    unit_test = code_gen_chain.invoke({"context":"","messages" : messages})
    messages +=  [("assistant",f"{unit_test.prefix} \n Imports: {unit_test.imports} \n Code: {unit_test.code}")]
    code_gen_result.set_latest_block("unit_test_code", unit_test.imports +"\n" + unit_test.code )
    iterations = iterations + 1
    return {"generation": unit_test, "messages": messages, "iterations": iterations, "error": "no"}

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
    # write the code to a file
    with open("temp.py", "w") as file:
        file.write(imports)
        file.write("\n")
        file.write(code)
    # Check imports
    try:
        exec(imports)
    except Exception as e:
        print("---CODE IMPORT CHECK: FAILED---")
        print(f"{e}")
        error_message = [("user", f"Your solution failed the import test: {e}")]
        messages += error_message
        return {"generation": code_solution, "messages": messages, "iterations": iterations, "error": "yes"}
    
    # Check execution
    try:
        exec(imports + "\n" + code)
    except Exception as e:
        print("---CODE BLOCK CHECK: FAILED---")
        print(f"solution failed the code execution test: {e}")
        error_message = [("user", f"Your solution failed the code execution test: {e}")]
        messages += error_message
        return {"generation": code_solution, "messages": messages, "iterations": iterations, "error": "yes"}
  
    # No errors
    print("---NO CODE TEST FAILURES---")
    return {"generation": code_solution, "messages": messages, "iterations": iterations, "error": "no"}


def unit_test_check(state: GraphState):
    print("---UNIT TEST CHECK---")
    
    messages = state["messages"]
    unit_test_code = state["generation"]
    iterations = state["iterations"]

    try:
        exec(code_gen_result.return_excutable_block(-1))
    except Exception as e:
        print("---UNIT TEST FAILED---")
        print(f"unit test failed: {e}")
        error_message = [("user", f"Your solution failed the unit test: {e}")]
        messages += error_message
        return {"generation": unit_test_code, "messages": messages, "iterations": iterations, "error": "yes"}
    
    print("---UNIT TEST PASSED---")
    return {"generation": unit_test_code, "messages": messages, "iterations": iterations, "error": "no"}
    

def reflect(state: GraphState):
    """
    Reflect on errors


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
    reflections = code_gen_chain.invoke({"context":"", "messages" : messages})
    messages += [("assistant" , f"Here are reflections on the error: {reflections}")]
    return {"generation": code_solution, "messages": messages, "iterations": iterations}

### Edges

def decide_to_finish_generate(state: GraphState):
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
        state["iterations"] = 0
        print("---DECISION: FINISH GENERATION---")
        if unit_flag:
            return "unit_test"
        else:
            return "end"
    else:
        print("---DECISION: RE-TRY SOLUTION---")
        if flag == "reflect":
            return "reflect"
        else:
            return "generate"


def decide_to_finish_unit_test(state: GraphState):
    """
    Determines whether to finish.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    error = state["error"]
    iterations = state["iterations"]

    if error == "no" or iterations == max_unit_iterations:
        print("---DECISION: FINISH---")
        code_gen_result.finish_block()
        return "end"
    else:
        print("---DECISION: RE-TRY SOLUTION---")
        if flag == 'reflect':
            return "reflect"
        else:
            return "unit_test"

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

def predict_langgraph(app, example: dict):
    """ LangGraph """
    graph = app.invoke({"context":"","messages":[("user",example["question"])],"iterations":0})
    solution = graph["generation"]
    return {"imports": solution.imports, "code": solution.code}

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("generate", generate)  # generation solution
workflow.add_node("check_code", code_check)  # check code
workflow.add_node("unit_test", generate_unit_tests)  # unit test
# workflow.add_node("reflect", reflect)  # reflect
workflow.add_node("unit_test_check", unit_test_check)  # unit test check

# Build graph
workflow.set_entry_point("generate")
workflow.add_edge("generate", "check_code")
workflow.add_edge("unit_test", "unit_test_check")
workflow.add_conditional_edges(
    "check_code",
    decide_to_finish_generate,
    {
        "end": END,
        "unit_test": "unit_test",
        "generate": "generate",
    },
)

workflow.add_conditional_edges(
    "unit_test_check",
    decide_to_finish_unit_test,
    {
        "end": END,
        "unit_test": "unit_test",
        # "generate": "generate",
    },
)
app = workflow.compile()
# Evaluator
code_evalulator = [check_import,check_execution]
input_json_path = "./dataloader.json"
import json
with open(input_json_path, "r") as file:
    inputs = json.load(file)
for i in range(len(inputs)):
    if "user_inputs" in inputs[i]:
        code_gen_result.user_inputs.append(inputs[i]["user_inputs"])
    else:
        code_gen_result.user_inputs.append("")
result = predict_langgraph(app, {"question": inputs[0]["question"]})
print(result)
code_gen_result.save_all_files("tempcode")

