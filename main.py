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
from typing import Any
#loading 
load_dotenv()

logging.basicConfig(level=logging.INFO)
#langgraph state
class Agent(TypedDict):
    query_human: str
    intent: dict
    repo: dict
    enrich_repos: list[dict]
    documents: list
    ranked_repos: list
    matching_repo_file:list[Document]
    chunked_repos:list[Document]
    vector_store:Any
    results:list[tuple[Document,float]]
    final_results:list[tuple]

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

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


#nodes - function


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
    try:
        response = model.invoke(prompt)
        state["intent"] = json.loads(response.content)
    except Exception as e:
        logging.error("Intent classification failed: %s", e)
        state["intent"] = {}
    return state




def finding_repos(state:Agent) -> Agent:
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
        query={
            "q":query_para,
            "sort":"stars",
            "order":"desc",
            "per_page":50,
            "page":1
        }

        response=requests.get("https://api.github.com/search/repositories", params=query, headers=headers)
        response.raise_for_status()
        result=response.json()
        state["repo"]=result.get("items", [])
        logging.info("GitHub items count: %d", len(state["repo"]))
    except Exception as e:
        logging.error("Finding repos failed: %s", e)
        state["repo"] = []
    return state


def normalizing_repo(state:Agent):
    try:
        all_repo=state.get("repo", [])
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
    except Exception as e:
        logging.error("Normalizing repo failed: %s", e)
        state["enrich_repos"] = []
    return state



def fetch_blob_content(owner, repo, blob_sha):
    url = f"https://api.github.com/repos/{owner}/{repo}/git/blobs/{blob_sha}"
    r = requests.get(url, headers=headers)
    r.raise_for_status()

    data = r.json()
    return base64.b64decode(data["content"]).decode("utf-8", errors="ignore")



def taking_files_from_repos(state:Agent) -> Agent:
    url=[]
    for each_url in state["enrich_repos"]:
        url.append(each_url["link"])

    listt_main_urls=[]
    for each_word in url:
        words = each_word.split("/")
        owner = words[-2]
        repo = words[-1]
        joining = f"/{owner}/{repo}/"
        main_code_directory_url = (
            f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1")
        listt_main_urls.append(main_code_directory_url)

    final_lang_ext = lang_ext.get(
    state["intent"]["github_search"]["language"])


    matching_files=[]
    for each in listt_main_urls:
        response=requests.get(url=each,headers=headers)
        results=response.json()

        parts = each.split("/")
        owner = parts[4]
        repo = parts[5]

        for e in results["tree"]:
            if e["type"]=="blob" and e["path"].endswith(final_lang_ext):
                matching_files.append(Document(
                    page_content=fetch_blob_content(owner,repo,e["sha"]),
                    metadata={
                        "repo": f"{owner}/{repo}",
                        "path": e["path"]
                    }
                ))


    state["matching_repo_file"]=matching_files
    return state
    
def chunking_repo(state:Agent) -> Agent:
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=[
        "\nclass ",
        "\ndef ",
        "\nasync def ",
        "\n\n",
        "\n",
        " "
    ]
)

    ingesting_files=state["matching_repo_files"]
    chunked_repos=[]
    for each_files in ingesting_files:
        chunks=splitter.split_text(each_files.page_content)
        for each_chunk in chunks:
            prompt=f'''
You are a senior software engineer.

Given the following code snippet, explain what it does in clear, concise terms.
Focus on:
- the purpose of the code
- the main responsibility of this logic
- how it fits into a larger system

Do NOT restate the code line by line.
Do NOT include implementation details unless necessary.
Write 2–3 sentences maximum.

Code:
{each_chunk}'''
            llm_response=model.invoke(prompt)
            chunked_repos.append(Document(
                page_content=llm_response.content,
                metadata=each_files.metadata
            ))
    state["chunked_repos"]=chunked_repos
    return state

def semantic_search(state:Agent) -> Agent:
    chunks=state["chunked_repos"]

    if "vector_store" not in state:
        state["vector_store"] = FAISS.from_documents(
            state["chunked_repos"],
            embedding
        )
    intent_text = " ".join([
            state["intent"].get("role", ""),
            state["intent"].get("seniority", ""),
            " ".join(state["intent"].get("primary_stack", [])),
            " ".join(state["intent"].get("domains", []))
        ])
    results=state["vector_store"].similarity_search_with_score(intent_text,k=10)
    state["results"]=results
    return state

def refined_results(state:Agent) -> Agent:
    h={}
    resultss=state["results"]
    for i in resultss:
        if i[0].metadata["repo"] in h:
            h[i[0].metadata["repo"]]+=1
        else:
            h[i[0].metadata["repo"]]=1
    results_refined=list(h.items())
    sorted_results_refined=sorted(results_refined,key=lambda x:x[1], reverse=True)
    state["final_results"]=sorted_results_refined
    return state
#graph 

graph=StateGraph(Agent)
graph.add_node("intent_classifier",intent_classifier)
graph.add_node("finding_repos",finding_repos)
graph.add_node("normalizing",normalizing_repo)
graph.add_node("files_ingest",taking_files_from_repos)
graph.add_node("chunking",chunking_repo)
graph.add_node("semantic",semantic_search)
graph.add_node("ranking_system",refined_results)

graph.add_edge(START,"intent_classifier")
graph.add_edge("intent_classifier","finding_repos")
graph.add_edge("finding_repos","normalizing")
graph.add_edge("normalizing", "files_ingest")
graph.add_edge("files_ingest", "chunking")
graph.add_edge("chunking", "semantic")
graph.add_edge("semantic","ranking_system")
graph.add_edge("ranking_system",END)

app=graph.compile()
result = app.invoke({
    "query_human": "find ai engineer mastered in langchain and langraph"
})

print(result["final_results"])

#dont run this, we are currently working on it :)
#and also in this code i am hitting rate limits on github api dude to many blobs , would love if anyone could  help me !