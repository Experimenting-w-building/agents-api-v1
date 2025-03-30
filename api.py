from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from github_rag import GitHubRAGSystem
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
import json
from langchain_deepseek import ChatDeepSeek
from langchain.output_parsers import OutputFixingParser
from typing import List
from fastapi import UploadFile, File
from langchain.schema import Document
import os

class Subheading(BaseModel):
    title: str

class Heading(BaseModel):
    title: str
    subheadings: List[Subheading]

class DocumentationStructure(BaseModel):
    headings: List[Heading]

app = FastAPI()
github_token = os.getenv("GITHUB_TOKEN")
rag = GitHubRAGSystem(github_token)
qa_chains = GitHubRAGSystem.init_vector_dbs(rag)

# HELPERS (start)
async def _upload_repo(rag_system: GitHubRAGSystem, repo):
    documents = await rag_system.fetch_github_data(repo)
    chunks = rag_system.chunk_documents(documents)
    vector_store = rag_system.initialize_vector_store(chunks)
    qa_chains[repo.replace("/", "-")] = rag_system.create_qa_chain(vector_store)
# HELPERS (end)

class UploadRequest(BaseModel):
    gh_token: Optional[str] = None
    repo: str

class QueryReqest(BaseModel):
    query: str

class DocContentRequest(BaseModel):
    heading: str
    subheading: str

@app.post("/")
async def upload_repo(req: UploadRequest):
    try:
        rag_system = GitHubRAGSystem(req.gh_token or github_token)
        await _upload_repo(rag_system, req.repo)
        return f'Uploaded {req.repo}'
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal server error occured.")

@app.post("/{repo_name}")
async def invoke_qa_chain(repo_name: str, req: QueryReqest):
    if repo_name in qa_chains:
        qa_chain = qa_chains[repo_name]
        result = qa_chain.invoke({"query": req.query})
        return {"result": result["result"], "sources": result["source_documents"]}
    else:
        raise HTTPException(status_code=404, detail="Repo not found")
    
@app.get("/all-agents")
async def get_all_agents():
    return {"result": rag.get_all_agents()}
    
@app.post("/{repo_name}/upload_docs")
async def upload_docs(repo_name: str, files: List[UploadFile] = File(...)):
    try:
        rag_system = GitHubRAGSystem(github_token=github_token, repo_name=repo_name)
        documents = []

        for file in files:
            if file.filename.endswith('.md'):
                content = await file.read()
                documents.append(Document(content.decode('utf-8')))  # Decode and append the content

        # Create embeddings directly from the documents
        vector_store = rag_system.initialize_vector_store(documents)
        qa_chains[repo_name.replace("/", "-")] = rag_system.create_qa_chain(vector_store)

        return f'Uploaded {len(documents)} Markdown files to {repo_name}'
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal server error occurred.")


@app.get("/{repo_name}/docs-structure")
async def get_docs_structure(repo_name: str):
    qdrant_client = QdrantClient("http://localhost:6333")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = Qdrant(client=qdrant_client, collection_name=repo_name, embeddings=embeddings)
    llm=ChatOpenAI(temperature=0, model="gpt-4o-mini")
    base_parser = JsonOutputParser(pydantic_object=DocumentationStructure)

    parser = OutputFixingParser.from_llm(
        parser=base_parser,
        llm=llm
    )

    data = {
        "headings": [
            {
                "title": "Main Heading",
                "subheadings": ["Subheading 1", "Subheading 2"]
            }
        ]
    }

    format_one_line = json.dumps(data, separators=(",", ":"))

    if repo_name in qa_chains:
        prompt_template = """Analyze the following GitHub repository content to create a structure for documentation.
        Consider code files, pull requests, and commits in your response, but do not include anything else but the structure
        for documentation in your answer. 
        Also, DO NOT speak of the context provided, but instead just use it to create a documentation structure naturally.
        DO NOT CREATE FAQs.

        Context:
        {context}

        Provide a detailed structure for documentation in JSON format with the key 'headings'.
        The JSON should look like this:
        {format_instructions}
        Answer:"""
        
        QA_PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context"],
            partial_variables={"format_instructions": format_one_line}
        )

        doc_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0.7, model="gpt-4o-mini"),
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 10}),
            chain_type_kwargs={"prompt": QA_PROMPT}
        )

        doc_result = doc_chain.invoke({
            "query": "Create a documentation structure for this repository.",
        })

        try:
            output = parser.parse(doc_result["result"])
            return {"result": output}
        except Exception as e:
            print(f"Output parsing failed: {e}")
            return {"error": "Failed to parse output", "details": str(e)}
            
    else:
        raise HTTPException(status_code=404, detail="Repo not found")
    

@app.get("/{repo_name}/docs-content")
async def get_docs_content(repo_name: str, req: DocContentRequest):
    qdrant_client = QdrantClient("http://localhost:6333")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = Qdrant(client=qdrant_client, collection_name=repo_name, embeddings=embeddings)
    llm=ChatDeepSeek(temperature=0, model="deepseek-reasoner")

    if repo_name in qa_chains:
        prompt_template = """
        **Description:** You are an AI system designed to generate structured documentation in **Markdown format**. 
        The documentation follows a predefined structure with headings and subheadings. 
        Given a heading or subheading, your task is to generate a detailed documentation file based on the context given with the following requirements:  

        ### Instructions:
        1. **Content Generation:**
        - Write clear, concise, and technical content based on the provided heading or subheading.
        - Include relevant explanations, examples, and code snippets where necessary.
        - If applicable, generate **Mermaid diagrams** to visualize workflows, architecture, or sequences.
        
        2. **Markdown Formatting:**
        - Use proper Markdown syntax for headings, lists, tables, and code blocks.
        - Include internal links where relevant for cross-referencing sections.
        - Add comments or notes in `> [!NOTE]` blocks if extra clarification is needed.

        3. **Structure:**
        - Start with an introductory paragraph explaining the section's purpose.
        - Provide detailed technical information or instructions.
        - Conclude with a summary or best practices.

        4. **Mermaid Diagrams (if necessary):**
        - Use Mermaid code blocks with proper syntax.
        - Label diagrams and provide brief descriptions.

        Example Mermaid diagram:  
        User query flow:  

        ```mermaid
        graph TD;
            A[User Query] --> B[Retrieval System]
            B --> C[Document Embeddings]
            C --> D[Contextual Search]
        ```
        Also, DO NOT speak of the context provided but USE it to create a proper documentation page.
        THE DOCUMENTATION SHOULD BE ABOUT THE REPOSITORY!
        
        Context:
        {context}

        Heading:
        {heading}

        Subheading:
        {subheading}

        Answer:"""
        
        QA_PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context"],
            partial_variables={
                "heading": req.heading,
                "subheading": req.subheading
            }
        )

        doc_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True
        )

        doc_result = doc_chain.invoke({
            "query": f"Provide detailed documentation for the heading '{req.heading}' and subheading '{req.subheading}'."
        })

        print(doc_result)

        return {
                "result": doc_result["result"],
            }
            
    else:
        raise HTTPException(status_code=404, detail="Repo not found")