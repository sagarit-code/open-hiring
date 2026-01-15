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



load_dotenv()


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
    api_key=os.getenv("GROQ_API_KEY")
)

headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "langgraph-agent"
    }


def intent_classifier(state: Agent) -> Agent:
    intent_schema= """
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

    response = model.invoke(prompt)
    state["intent"] = json.loads(response.content)
    return state


    response=model.invoke(prompt)
    state["intent"] = json.loads(response.content)

    return state

def finding_repos(state:Agent) -> Agent:
    url="https://api.github.com/search/repositories"
    query_para = " ".join([
    f"language:{state["intent"]["github_search"]["language"]}",
    f"topic:{state["intent"]["github_search"]["topics"][0]}",
    f"topic:{state["intent"]["github_search"]["topics"][1]}",
    "stars:>50",
    "forks:>5",
    "fork:false"
])

    query={
        "q":query_para,
        "sort":"stars",
        "order":"desc",
        "per_page":50,
        "page":1
    }

    response=requests.get(url=url,params=query,headers=headers)
    result=response.json()

    state["repo"]=result["items"]
    print("GitHub items count:", len(result.get("items", [])))

    return state

def normalizing_repo(state:Agent):
    all_repo=state["repo"]
    quality_leads=[]
    
    for each in all_repo:
        if each["owner"]["type"] in ["User", "Organization"]:
            info={
                "user": each["owner"]["login"],
                "repo": each["name"],
                "link":each["html_url"],
                "stars":each["stargazers_count"],
                "forks": each["forks_count"],
                "language": each["language"],
                "pushed_at": each["pushed_at"],
                }
            quality_leads.append(info)
    state["enrich_repos"]=quality_leads
    

    return state

def fetch_readme(owner: str, repo: str) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    res = requests.get(url, headers=headers)

    if res.status_code != 200:
        return ""

    data = res.json()
    content = data.get("content", "")
    if not content:
        return ""

    return base64.b64decode(content).decode("utf-8", errors="ignore")

def build_documents(state: Agent) -> Agent:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    docs = []

    for repo in state["enrich_repos"]:
        readme = fetch_readme(repo["user"], repo["repo"])
        if not readme:
            continue

        chunks = splitter.split_text(readme)

        for chunk in chunks:
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "repo": repo["repo"],
                        "owner": repo["user"],
                        "stars": repo["stars"],
                        "url": repo["link"]
                    }
                )
            )

    state["documents"] = docs
    print("Total chunks:", len(docs))
    return state


def semantic_rank(state: Agent) -> Agent:
    if not state["documents"]:
        state["ranked_repos"] = []
        return state

    vectordb = FAISS.from_documents(
        state["documents"],
        embedding
    )

    intent_text = " ".join([
        state["intent"]["role"],
        state["intent"]["seniority"],
        " ".join(state["intent"]["primary_stack"]),
        " ".join(state["intent"]["domains"])
    ])

    results = vectordb.similarity_search(
        intent_text,
        k=10
    )

    repo_scores = {}

    for doc in results:
        key = f"{doc.metadata['owner']}/{doc.metadata['repo']}"
        repo_scores[key] = {
            "repo": key,
            "url": doc.metadata["url"],
            "stars": doc.metadata["stars"],
            "match_excerpt": doc.page_content[:300]
        }

    state["ranked_repos"] = list(repo_scores.values())
    return state



graph=StateGraph(Agent)
graph.add_node("intent_classifier",intent_classifier)
graph.add_node("finding_repos",finding_repos)
graph.add_node("normalizing",normalizing_repo)
graph.add_node("build_documents", build_documents)
graph.add_node("semantic_rank", semantic_rank)



graph.add_edge(START,"intent_classifier")
graph.add_edge("intent_classifier","finding_repos")
graph.add_edge("finding_repos","normalizing")
graph.add_edge("normalizing", "build_documents")
graph.add_edge("build_documents", "semantic_rank")
graph.add_edge("semantic_rank", END)

app=graph.compile()
result = app.invoke({
    "query_human": "find ai engineer mastered  in langchain and langraph"
})

print("\nTOP MATCHED REPOS:\n")
for r in result["ranked_repos"]:
    print(r["url"], "->", r["stars"])
