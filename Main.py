import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")

from fastapi import FastAPI, HTTPException
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel
from dotenv import load_dotenv
import json
import re
import os


app = FastAPI(title = "Infrascribe Agent")
load_dotenv()

llm = ChatGroq(model = "llama-3.3-70b-versatile")
# class ConfigurableField(BaseModel):
#     key: str
#     label: str
#     value: Any  # could be str, int, bool, list
#     input_type: str  # "text", "select", "number", "toggle"
#     options: list[Any] = []  # empty if not a select

# class Component(BaseModel):
#     id: str
#     name: str
#     type: str  # e.g. "aws_instance", "aws_db_instance"
#     description: str
#     configurable_fields: list[ConfigurableField]

# class CostBreakdown(BaseModel):
#     component_id: str
#     name: str
#     monthly: float

# class Cost(BaseModel):
#     total_monthly: float
#     breakdown: list[CostBreakdown]
#     notes: str

# class InfrastructureResponse(BaseModel):
#     summary: str
#     chat_message: str
#     components: list[Component]  # dynamic — Claude decides how many and what type
#     terraform: str
#     cost: Cost

class ProjectRequest(BaseModel):
    description: str

class ProjectResponse(BaseModel):
    summary: str
    terraform: str
    estimated_cost: str
    alternatives: str


system_prompt = """
    You are an expert AWS Cloud Architect and Terraform engineer. You will consult with a user on what Infrastructure is needed to run their project and the total cost of their infrastructure and 
    offer better alternatives.

    Answer the user query and use necessary tools.

    When given a project description, you will:
    1. Identify the required AWS infrastructure components
    2. Generate a complete, valid Terraform HCL template
    3. Return a structured JSON response

    When given a project description, respond ONLY in raw JSON with no markdown, no code fences, no extra text:
    {
        "summary": "brief summary of the infrastructure",
        "terraform": "the complete terraform HCL code with \\n for newlines",
        "estimated_cost": "estimated monthly cost"
        "alternatives": "suggest better alternatives"
    }

    Terraform requirements:
    - Always include a provider block for AWS with region variable
    - Use sensible defaults and free-tier friendly options where possible
    - Include comments in the HCL explaining each resource
    - Use variables for region, instance type, and other configurable values
    - Always include an outputs block
    """

# --- Endpoints ---
@app.post("/generate", response_model=dict)
async def generate_infrastructure(request: ProjectRequest):
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Project Description: {request.description}")
        ]
        
        response = llm.invoke(messages)
        clean = re.sub(r"```(?:json)?", "", response.content).strip()
        parsed = json.loads(clean, strict=False)
        validated = ProjectResponse(**parsed)

        os.makedirs("terraform_outputs", exist_ok=True)
        with open("terraform_outputs/main.tf","w") as f:
            f.write(validated.terraform)

        return validated.model_dump()
    
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"JSON parse error at pos {e.pos}: {e.msg}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
