from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional
import os
from dotenv import load_dotenv
import json
import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import base64
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Loading
load_dotenv()

logging.basicConfig(level=logging.INFO)

# FastAPI app
app = FastAPI(title="OpenHiring API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class SearchRequest(BaseModel):
    query: str

# langgraph state
class Agent(TypedDict):
    query_human: str
    intent: dict
    repo: dict
    enrich_repos: list[dict]
    documents: list
    ranked_repos: list

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load intent examples
with open("intent_ex.json", "r", encoding="utf-8") as f:
    INTENT_EXAMPLES = json.load(f)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY
)

headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "User-Agent": "langgraph-agent"
}

lang_ext = {
    "Python": ".py",
    "JavaScript": ".js",
    "TypeScript": ".ts",
    "Go": ".go",
}


# nodes - function

def intent_classifier(state: Agent) -> Agent:
    intent_schema = """
{
  "role": "string",
  "seniority": "string",
  "primary_stack": ["string"],
  "secondary_stack": ["string"],
  "domains": ["string"],
  "evaluation_signals": ["string"],
  "github_search": {
    "language": "string",
    "topics": ["string", "string", "string"]
  }
}
"""
    query = state["query_human"]
    prompt = f"""
You are an intent extraction system.

Rules:
- Convert user hiring intent into structured JSON
- Follow the exact schema used in the examples
- Do not invent technologies
- Infer conservatively
- Output ONLY valid JSON
- Do NOT add explanations or comments

GitHub Search Constraints (IMPORTANT):
- github_search.language MUST be exactly ONE string
- github_search.language must be a programming language (e.g. "Python")
- github_search.topics MUST be an array with only 3 items
- Do NOT include more than 3 topics or less than 3
- If unsure, include similar topics

You MUST output JSON with this exact structure:
{intent_schema}

Canonical examples:
{json.dumps(INTENT_EXAMPLES, indent=2)}

User input:
{query}
"""
    try:
        response = model.invoke(prompt)
        state["intent"] = json.loads(response.content)
    except Exception as e:
        logging.error("Intent classification failed: %s", e)
        state["intent"] = {}
    return state


def finding_repos(state: Agent) -> Agent:
    try:
        query_para = " ".join([
            f"language:{state['intent']['github_search']['language']}",
            f"topic:{state['intent']['github_search']['topics'][0]}",
            f"topic:{state['intent']['github_search']['topics'][1]}",
            f"topic:{state['intent']['github_search']['topics'][2]}",
            "stars:>50",
            "forks:>5",
            "fork:false"
        ])
        query = {
            "q": query_para,
            "sort": "stars",
            "order": "desc",
            "per_page": 50,
            "page": 1
        }

        response = requests.get("https://api.github.com/search/repositories", params=query, headers=headers)
        response.raise_for_status()
        result = response.json()
        state["repo"] = result.get("items", [])
        logging.info("GitHub items count: %d", len(state["repo"]))
    except Exception as e:
        logging.error("Finding repos failed: %s", e)
        state["repo"] = []
    return state


def normalizing_repo(state: Agent):
    try:
        all_repo = state.get("repo", [])
        quality_leads = []
        for each in all_repo:
            if each["owner"]["type"] in ["User", "Organization"]:
                info = {
                    "user": each["owner"]["login"],
                    "repo": each["name"],
                    "url": each["html_url"],
                    "link": each["html_url"],
                    "stars": each["stargazers_count"],
                    "forks": each["forks_count"],
                    "language": each["language"],
                    "pushed_at": each["pushed_at"],
                }
                quality_leads.append(info)
        state["enrich_repos"] = quality_leads
        # For now, just use enrich_repos as ranked_repos
        # You can add your semantic ranking logic later
        state["ranked_repos"] = quality_leads[:10]  # Top 10
    except Exception as e:
        logging.error("Normalizing repo failed: %s", e)
        state["enrich_repos"] = []
        state["ranked_repos"] = []
    return state


def fetch_blob_content(owner, repo, blob_sha):
    url = f"https://api.github.com/repos/{owner}/{repo}/git/blobs/{blob_sha}"
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    data = r.json()
    return base64.b64decode(data["content"]).decode("utf-8", errors="ignore")


def taking_files_from_repos(state: Agent) -> Agent:
    # This function is incomplete in your original code
    # I'll leave it as a placeholder for now
    state["documents"] = []
    return state


def build_documents(state: Agent) -> Agent:
    # Placeholder for document building
    state["documents"] = []
    return state


def semantic_rank(state: Agent) -> Agent:
    # Placeholder for semantic ranking
    # For now, just keep the normalized repos
    if "ranked_repos" not in state:
        state["ranked_repos"] = state.get("enrich_repos", [])
    return state


# graph
graph = StateGraph(Agent)
graph.add_node("intent_classifier", intent_classifier)
graph.add_node("finding_repos", finding_repos)
graph.add_node("normalizing", normalizing_repo)
graph.add_node("build_documents", build_documents)
graph.add_node("semantic_rank", semantic_rank)

graph.add_edge(START, "intent_classifier")
graph.add_edge("intent_classifier", "finding_repos")
graph.add_edge("finding_repos", "normalizing")
graph.add_edge("normalizing", "build_documents")
graph.add_edge("build_documents", "semantic_rank")
graph.add_edge("semantic_rank", END)

compiled_app = graph.compile()


# API Routes

@app.get("/")
async def root():
    return {"message": "OpenHiring API is running"}


@app.post("/search")
async def search_developers(request: SearchRequest):
    try:
        logging.info(f"Received search query: {request.query}")
        
        result = compiled_app.invoke({
            "query_human": request.query
        })
        
        return {
            "query": request.query,
            "intent": result.get("intent", {}),
            "ranked_repos": result.get("ranked_repos", [])
        }
    
    except Exception as e:
        logging.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("\n🚀 Starting OpenHiring API Server...")
    print("📍 API URL: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs")
    print("\nMake sure to:")
    print("1. Have your .env file with GITHUB_TOKEN and GROQ_API_KEY")
    print("2. Have intent_ex.json in the same directory")
    print("3. Open frontend/index.html in your browser\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)