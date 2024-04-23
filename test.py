
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from dotenv import load_dotenv
from numpy import block
load_dotenv(dotenv_path='.env', override=True)
import os 
from typing import Dict, TypedDict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import END, StateGraph
from langchain_core.pydantic_v1 import BaseModel, Field
import langsmith
import subprocess
from langsmith.schemas import Example, Run
import tempfile
### Parameter
# Max tries
max_iterations = 3
max_unit_iterations = 3
max_block_iterations = 3

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
    [("system","""You are a coding asistant to generate function code and test code. The user will give specific task descriptions. Just give fully executable end to end code. Nothing should be missing."""),
    ("placeholder", "{messages}")]
)

global_gen_prompt = ChatPromptTemplate.from_messages(
    [("system", """You are a world-class programmer and AI assistant capable of executing any goal related to software development, genAI, LLMs, and full-stack technologies. Write a code for a main script based on the user prompt. The user prompt is [user prompt]: {user_prompt}. Now taking user_prompt, I want you to first split it into subprompts and figure out which subprompts can be parallelized and which are sequential. Give me end to end code structure and the code to execute these subprompts parallely or sequentially based on what you decided. Now for each sub prompt, I have a function called systemcheck which takes in the input of the subprompt. Now, show me the code for how you are parallelizing this and how you are making it sequential clearly. Include all the implementation details. You are free to make decisions as to what the CNN Architecture is, or which performance metrics to choose etc. Just give fully executable end to end code. Nothing should be missing.""")]
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
    code: str = Field(description="Code block not including import statements. It has the function code for the problem, providing a funciton to use.")
    # code: str = Field(description="Code block for unit tests, only the test code to check the function")
    description = "Schema for code solutions to questions."


class main_code(BaseModel):
    """Code output"""
    name: str = Field(description="Name of the code block")
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements. ")
    # code: str = Field(description="Code block for unit tests, only the test code to check the function")
    description = "Schema for code solutions to questions."
class CodeBlock:
    def __init__(self) -> None:
        self.prefix = ""
        self.imports = ""
        self.code = ""
        self.test_code = ""
        self.name = ""
        self.description = ""

        
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
            return "\n".join([block.test_code])
        else:
            return "\n".join([ block.imports + "\n" + block.code])
        # return "\n".join([ block.imports + "\n" + block.code + "\n" + block.unit_test_code])
    
    def get_latest_user_input(self):
        return self.user_inputs[len(self.blocks) - 1]
    
    def save_all_files(self, dir_name="tempcode", file_name=None):
        os.makedirs(dir_name, exist_ok=True)
        for i, block in enumerate(self.blocks[:-1]):
            if file_name is None:
                file_name = block.name
            with open(f"{dir_name}/{file_name}.py", "w") as file:
                file.write(self.return_excutable_block(i))
    

code_gen_chain = code_gen_prompt | llm.with_structured_output(code)
global_gen_chain = global_gen_prompt | llm.with_structured_output(main_code )

code_gen_result = CodeBlocks()

def write_code_to_file(code: str, file_name: str):
    with open("tempcode/"+file_name+".py", "w") as file:
        file.write(code)
        
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
    block_iterations : int
    question : str


def excute_code(code: str):
    with tempfile.NamedTemporaryFile(mode='w', suffix=".py") as file:
        file.write(code)
        file.seek(0)
        output = subprocess.run(['python', file.name], capture_output=True, text=True)
    return output.stderr

### Nodes
def regenerate(state: GraphState):
    state.update({"block_iterations": state["block_iterations"] + 1, "iterations": 0})
    return state

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
        messages += [("user","Now, try again. Invoke the code tool to structure the output with a prefix, imports, function code block.")]
        
    # Solution
    code_solution = code_gen_chain.invoke({"context":"","messages" : messages})
    code_gen_result.set_latest_block("prefix", code_solution.prefix)
    code_gen_result.set_latest_block("imports", code_solution.imports)
    code_gen_result.set_latest_block("code", code_solution.code)
    # code_gen_result.set_latest_block("test_code", code_solution.test_code)
    code_gen_result.set_latest_block("name", code_solution.name)
    code_gen_result.set_latest_block("description", code_solution.description)

    write_code_to_file(code_solution.imports + "\n" + code_solution.code, "1")
    messages += [("assistant",f"Function Code: {code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}")]
    request_for_test_code = [("user", "Now, provide the test code to test the function with randomly generalized input")]
    messages += request_for_test_code
    
    
    print("---GENERATING TEST CODE SOLUTION---")
    test_code_solution = code_gen_chain.invoke({"context":"","messages" : messages})
    
    test_imports = test_code_solution.imports
    test_code_block = test_code_solution.code
    
    messages += [("assistant",f"Test Code: Imports: {test_imports} \n Code: {test_code_block}")]
    concat_code = "\n".join([code_solution.imports, code_solution.code, test_code_solution.imports, test_code_solution.code])
    write_code_to_file(concat_code, "2")
    request_for_code_clean = [("user", f"this is the concated code, including the function code and test code. Please clean the code and provide the final version. Remeber to call the test function in main:\n {concat_code}")]
    # Increment
    messages += request_for_code_clean
    
    print("---GENERATING FINAL CODE SOLUTION---")
    final_code_solution = code_gen_chain.invoke({"context":"","messages" : messages})
    code_gen_result.set_latest_block("code", final_code_solution.code)
    code_gen_result.set_latest_block("imports", final_code_solution.imports)
    write_code_to_file(final_code_solution.imports + "\n" + final_code_solution.code, "3")
    iterations = iterations + 1
    state.update({"generation": final_code_solution, "messages": messages, "iterations": iterations})
    print(state.keys())
    return state


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
    # prefix = code_solution.prefix
    block_imports = code_solution.imports
    block_code = code_solution.code
    # write the code to a file
    e = excute_code(block_imports)
    if e:
        print("---CODE IMPORT CHECK: FAILED---")
        print(f"{e}")
        error_message = [("user", f"Your solution failed the import test: {e}")]
        messages += error_message
        state.update({"generation": code_solution, "messages": messages, "iterations": iterations, "error": "yes"})
        return state
    
    code = block_imports + "\n" + block_code

    e = excute_code(code)
    if e:
        print("---CODE BLOCK CHECK: FAILED---")
        print(f"solution failed the code execution test: {e}")
        error_message = [("user", f"Your solution failed the function code execution test: {e}")]
        messages += error_message
        state.update({"generation": code_solution, "messages": messages, "iterations": iterations, "error": "yes"})
        return state


    
    # No errors
    print("---NO CODE TEST FAILURES---")
    state.update({"generation": code_solution, "messages": messages, "iterations": iterations, "error": "no"})
    return state



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
        
        print("---DECISION: FINISH GENERATION---")
        code_gen_result.finish_block()
        return "end"
    else:
        print(f"---DECISION: RE-TRY SOLUTION---, iterations:{iterations}")
        if flag == "reflect":
            return "reflect"
        else:
            return "generate"


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
    graph = app.invoke({"question":example["question"],"messages":[("user",example["question"])],"iterations":0, "block_iterations":0, "question": example["question"]})
    solution = graph["generation"]
    return {"imports": solution.imports, "code": solution.code}

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("generate", generate)  # generation solution
workflow.add_node("check_code", code_check)  # check code

# Build graph
workflow.set_entry_point("generate")
workflow.add_edge("generate", "check_code")

workflow.add_conditional_edges(
    "check_code",
    decide_to_finish_generate,
    {
        "end": END,
        "generate": "generate",
    },
)


app = workflow.compile()
# Evaluator
code_evalulator = [check_import,check_execution]
input_json_path = "/u/jiahuad2/codebase/hack/workflow-agent/anqi.json"
import json
with open(input_json_path, "r") as file:
    inputs = json.load(file)
for i in range(len(inputs)):
    if "user_inputs" in inputs[i]:
        code_gen_result.user_inputs.append(inputs[i]["user_inputs"])
    else:
        code_gen_result.user_inputs.append("")
        
with open("global.txt", "r") as file:
    global_descriptions = "".join(file.readlines())

global_response = global_gen_chain.invoke({"user_prompt" : global_descriptions})
write_code_to_file(global_response.imports + "\n" + global_response.code, "main")
exit(0)
result = predict_langgraph(app, {"question": inputs[0]["question"]})
print(result)
code_gen_result.save_all_files("tempcode")

