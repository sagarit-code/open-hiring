
# 🔍 Semantic GitHub Talent Discovery Engine

> Find real developers from real code — not resumes, not keywords, not noise.

This project is an **AI-powered talent discovery engine** that searches GitHub, understands developer intent, reads README files, and **semantically ranks repositories (and developers)** based on how well they match a hiring query.

No scraping profiles.
No guessing skills.
Just **signal from code**.

---

## ✨ What This Does 

You give it a query like:

> *“Find me a backend developer experienced in FastAPI”*

The system will:

1. **Understand the intent** using an LLM
2. **Search GitHub** with strict, high-signal filters
3. **Normalize raw GitHub data** (owners, repos, stars, forks, activity)
4. **Fetch README files** (where the real story lives)
5. **Chunk READMEs intelligently**
6. **Generate embeddings** for semantic meaning
7. **Perform vector search + re-ranking**
8. Return the **most relevant repositories** (and implicitly, developers)

This is not keyword matching.
This is **semantic matching on real work**.

---

## 🧠 Why This Is Different

Most tools ask:

> “Does this profile mention FastAPI?”

This system asks:

> “Does this code actually *demonstrate* FastAPI experience in a meaningful way?”

### Key differences:

* ✅ Uses **code + README**, not bios
* ✅ Schema-locked intent extraction (no LLM hallucinations)
* ✅ Vector search over real technical content
* ✅ Works even when repo descriptions are empty
* ✅ Built for **real-world hiring signals**

---

## 🏗️ Architecture Overview

```
User Query
   ↓
Intent Classifier (LLM → Structured JSON)
   ↓
GitHub Search (strict filters, no noise)
   ↓
Repo Normalization
   ↓
README Fetching
   ↓
Text Chunking
   ↓
Embeddings
   ↓
Vector Store
   ↓
Semantic Ranking
   ↓
Top Matched Repositories
```

Powered by:

* **LangGraph** – deterministic agent workflows
* **LangChain** – chunking, embeddings, retrieval
* **Groq (LLaMA 3.1)** – fast intent extraction
* **GitHub API** – trusted source of truth

---

## 🧩 Example Output

```
TOP MATCHED REPOS:

https://github.com/kennethleungty/Llama-2-Open-Source-LLM-CPU-Inference ⭐ 972
https://github.com/zhongyao/openchat                         ⭐ 464
https://github.com/zilliztech/akcio                          ⭐ 259
```

These aren’t random.
They’re **semantically aligned** with the intent.

---

## 🚀 Getting Started

### 1️⃣ Clone the repo

```bash
git clone https://github.com/your-username/semantic-github-talent-engine
cd semantic-github-talent-engine
```

### 2️⃣ Set environment variables

Create a `.env` file:

```env
GITHUB_TOKEN=your_github_token
GROQ_API_KEY=your_groq_api_key
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Install Frontend Server

```bash
npm install -g http-server
```

### 5️⃣ Run the Application

Open two separate terminals:

**Terminal 1 (Backend):**
```bash
uvicorn main:app --reload
```
(This will start the API server at `http://localhost:8000`)

**Terminal 2 (Frontend):**
```bash
http-server frontend
```
(This will serve your frontend files, usually at `http://localhost:8080`)

Then, open your web browser and navigate to the frontend URL (e.g., `http://localhost:8080`).

Example query:

```python
result = app.invoke({
    "query_human": "find ai engineer mastering langchain and langgraph"
})
```

---

## 🧪 Current Capabilities

* ✔ Backend roles
* ✔ AI / ML engineers
* ✔ Open-source contributors
* ✔ Tool builders
* ✔ Infra & platform engineers

---

## 🛣️ Roadmap

* [ ] Aggregate scores per developer (not just repos)
* [ ] Multi-signal ranking (stars, recency, consistency)
* [ ] “Why matched” explanations (fully grounded, no hallucination)
* [ ] UI / dashboard
* [ ] Company vs individual profiling
* [ ] Hiring pipeline integration

---

## ⚠️ Philosophy

> **Code doesn’t lie. Resumes do.**

This project is built on the belief that:

* Real skill leaves artifacts
* Open-source is the strongest signal
* Semantics > keywords
* Deterministic pipelines > magic prompts

---

## 🧠 Inspiration

Inspired by systems like:

* SkillSync
* Eightfold
* SeekOut

…but built **open, explainable, and developer-first**.

---

## 🤝 Contributing

If you’re excited about:

* search systems
* embeddings
* RAG
* hiring tech
* open-source intelligence

PRs and discussions are welcome.

---


