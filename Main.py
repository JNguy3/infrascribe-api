import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel
from dotenv import load_dotenv
import json
import re
import os


app = FastAPI(title = "Infrascribe Agent")

origins = [
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
load_dotenv()

llm = ChatGroq(model = "llama-3.3-70b-versatile")

class ProjectRequest(BaseModel):
    description: str

class ProjectResponse(BaseModel):
    summary: str
    terraform: str
    estimated_cost: str
    alternatives: str

class AiIntroduction(BaseModel):
    introduction: str

introduction_prompt = """
    Your name is Infrascribe, expert AWS Cloud Architect and Terraform engineer. 

    Introduce yourself to the user as Infrascribe and only respond in Raw JSON with no markdown, no code fences, no extra text:
    {
        "introduction": "brief introduction about yourself and what you can do. End the introduction with 'How may I help you today?'"
    }
    """
system_prompt = """
    You are an expert AWS Cloud Architect and Terraform engineer. Your name is Infrascribe. You will consult with a user on what Infrastructure is needed to run their project and the total cost of their infrastructure and 
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

@app.get("/introduction", response_model=dict)
async def ai_introduction():
    try:
        introduction = [
            SystemMessage(content=introduction_prompt)
        ]
        response = llm.invoke(introduction)

        clean = re.sub(r"```(?:json)?", "", response.content).strip()
        parsed = json.loads(clean, strict=False)
        validated = AiIntroduction(**parsed)
        return validated.model_dump()
    
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"JSON parse error at pos {e.pos}: {e.msg}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))