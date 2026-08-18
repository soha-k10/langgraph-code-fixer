import os
import streamlit as st
from typing import TypedDict, List, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START

# Load environment variables
load_dotenv()

# ==========================================
# 1. Pydantic Schemas (Type-Safe Data Enforcer)
# ==========================================

class TestCase(BaseModel):
    input_args: Dict[str, Any] = Field(description="Dictionary of argument names and values, e.g., {'numbers': [1, 2]}")
    expected_output: Any = Field(description="Expected return value from function")
    description: str = Field(description="Reason for testing this specific input")

class TestSuiteSchema(BaseModel):
    test_cases: List[TestCase]

class CodePatch(BaseModel):
    fixed_code: str = Field(description="The executable patched Python code")
    explanation: str = Field(description="Clear breakdown of why the original code failed and how it was fixed")

# ==========================================
# 2. LangGraph State Definition
# ==========================================

class BugFixerState(TypedDict):
    current_code: str
    test_suite_json: str
    test_results: Dict[str, Any]
    patch_explanation: str
    attempts: int
    max_attempts: int

# ==========================================
# 3. LangGraph Node Functions
# ==========================================

def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API Key missing! Set it in .env or sidebar.")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=api_key)


def test_generator_node(state: BugFixerState) -> Dict[str, Any]:
    """Generates structured Pydantic test cases using LLM."""
    llm = get_llm()
    structured_llm = llm.with_structured_output(TestSuiteSchema)
    
    prompt = f"Generate 4 comprehensive test cases (including edge cases) for this Python function:\n\n{state['current_code']}"
    test_suite: TestSuiteSchema = structured_llm.invoke(prompt)
    
    return {"test_suite_json": test_suite.model_dump_json()}


def test_executor_node(state: BugFixerState) -> Dict[str, Any]:
    """Runs tests natively in Python memory and validates outputs using Pydantic structure."""
    test_suite = TestSuiteSchema.model_validate_json(state["test_suite_json"])
    
    # Safely execute user function in isolated namespace
    namespace = {}
    try:
        exec(state["current_code"], namespace)
        # Find defined function name
        func = [v for k, v in namespace.items() if callable(v)][0]
    except Exception as e:
        return {
            "test_results": {"passed": False, "output": f"Syntax/Compilation Error: {str(e)}"},
            "attempts": state["attempts"] + 1
        }
    
    failures = []
    
    for idx, test in enumerate(test_suite.test_cases, 1):
        try:
            result = func(**test.input_args)
            if result != test.expected_output:
                failures.append(f"Test {idx} ({test.description}): Expected {test.expected_output}, got {result}")
        except Exception as e:
            failures.append(f"Test {idx} ({test.description}): Unexpected crash -> {type(e).__name__}: {str(e)}")
            
    all_passed = len(failures) == 0
    output_log = "All Pydantic test cases passed successfully!" if all_passed else "\n".join(failures)
    
    return {
        "test_results": {"passed": all_passed, "output": output_log},
        "attempts": state["attempts"] + 1
    }


def patch_agent_node(state: BugFixerState) -> Dict[str, Any]:
    """Analyzes test failures and generates a fixed version with explanations."""
    llm = get_llm()
    structured_llm = llm.with_structured_output(CodePatch)
    
    prompt = (
        f"The following Python code failed test validation:\n\n{state['current_code']}\n\n"
        f"Failure Log:\n{state['test_results']['output']}\n\n"
        "Provide a fixed version of the code along with an explanation."
    )
    
    patch: CodePatch = structured_llm.invoke(prompt)
    
    # Remove any markdown code block formatting if present
    clean_code = patch.fixed_code.replace("```python", "").replace("```", "").strip()
    
    return {
        "current_code": clean_code,
        "patch_explanation": patch.explanation
    }


def should_continue(state: BugFixerState) -> str:
    if state["test_results"]["passed"]:
        return "passed"
    elif state["attempts"] >= state["max_attempts"]:
        return "max_attempts_reached"
    else:
        return "failed"

# ==========================================
# 4. Constructing the LangGraph
# ==========================================

@st.cache_resource
def build_graph():
    builder = StateGraph(BugFixerState)
    
    builder.add_node("generate_tests", test_generator_node)
    builder.add_node("execute_tests", test_executor_node)
    builder.add_node("patch_code", patch_agent_node)
    
    builder.add_edge(START, "generate_tests")
    builder.add_edge("generate_tests", "execute_tests")
    
    builder.add_conditional_edges(
        "execute_tests",
        should_continue,
        {
            "passed": END,
            "failed": "patch_code",
            "max_attempts_reached": END
        }
    )
    builder.add_edge("patch_code", "execute_tests")
    return builder.compile()

# ==========================================
# 5. Streamlit Frontend UI
# ==========================================

st.set_page_config(page_title="Autonomous Code Fixer", page_icon="🛠️", layout="wide")

st.title("🛠️ Autonomous Code Tester & Fixer")
st.caption("Powered by LangGraph, Pydantic, and OpenAI")

with st.sidebar:
    st.header("🔑 Configuration")
    user_api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    if user_api_key:
        os.environ["OPENAI_API_KEY"] = user_api_key

default_buggy_code = """def calculate_average(numbers):
    total = sum(numbers)
    return total / len(numbers)
"""

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Enter Python Code")
    input_code = st.text_area("Input Code", value=default_buggy_code, height=220)
    run_button = st.button("🚀 Test & Auto-Fix Code", type="primary", use_container_width=True)

with col2:
    st.subheader("2. Real-Time Fix Workflow")
    status_box = st.empty()

if run_button:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Please provide an OpenAI API Key in the sidebar or .env file.")
    else:
        app = build_graph()
        initial_state = {
            "current_code": input_code,
            "test_suite_json": "",
            "test_results": {},
            "patch_explanation": "",
            "attempts": 0,
            "max_attempts": 3
        }
        
        with st.spinner("Running LangGraph Autonomous Agent Loop..."):
            final_state = app.invoke(initial_state)
            
        if final_state["test_results"].get("passed"):
            st.success(f"✅ All tests passed in {final_state['attempts']} iteration(s)!")
        else:
            st.error("❌ Max fix attempts reached without passing all edge cases.")
            
        st.write("### 📝 Generated Pydantic Test Suite Cases:")
        if final_state.get("test_suite_json"):
            suite = TestSuiteSchema.model_validate_json(final_state["test_suite_json"])
            for t in suite.test_cases:
                st.write(f"- **Input:** `{t.input_args}` ➔ **Expected Output:** `{t.expected_output}` ({t.description})")
                
        st.write("### 🔍 Final Execution Results:")
        st.code(final_state["test_results"].get("output", ""), language="text")
        
        if final_state.get("patch_explanation"):
            st.write("### 💡 Root Cause & Patch Explanation:")
            st.info(final_state["patch_explanation"])
            
        st.write("### ⚡ Fixed Output Code:")
        st.code(final_state["current_code"], language="python")