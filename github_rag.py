from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import aiohttp
from typing import List

class GitHubRAGSystem:
    def __init__(self, github_token: str, repo_name: str = None):
        self.github_token = github_token
        self.headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.qdrant_client = QdrantClient("http://localhost:6333")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.repo = repo_name or ""
        
    async def fetch_github_data(self, repo: str) -> List[Document]:
        """Fetch repository content from GitHub API"""
        self.repo = repo.replace("/", "-")  # Convert repo name
        documents = []
        session = aiohttp.ClientSession(headers=self.headers)

        try:
            # Fetch repository metadata
            async with session.get(f'https://api.github.com/repos/{repo}') as response:
                if response.status != 200:
                    print(f"Error fetching repo info: {await response.text()}")
                    return []
                repo_data = await response.json()

            # Process repository content
            await self.process_directory(f'https://api.github.com/repos/{repo}/contents', session, documents)
            
            # Process pull requests
            pr_docs = await self.process_pulls(repo, session)
            documents.extend(pr_docs)
            
            # Process commits
            commit_docs = await self.process_commits(repo, session)
            documents.extend(commit_docs)

        except Exception as e:
            print(f"Error fetching GitHub data: {str(e)}")
        finally:
            await session.close()
            
        return documents

    def is_allowed_filetype(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        allowed_extensions = [
            '.py', '.txt', '.js', '.tsx', '.ts', '.md', 
            '.cjs', '.html', '.json', '.ipynb', '.h', 
            '.sh', '.yaml', '.java', '.cpp', '.go', '.rs'
        ]
        return any(filename.endswith(ext) for ext in allowed_extensions)

    async def process_directory(self, url: str, session: aiohttp.ClientSession, documents: List[Document]):
        """Recursively process repository directory"""
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    print(f"Error fetching directory {url}: {await response.text()}")
                    return
                
                contents = await response.json()
                if not isinstance(contents, list):
                    print(f"Unexpected response format from {url}")
                    return

            for item in contents:
                if not isinstance(item, dict):
                    continue
                    
                if item.get('type') == 'file':
                    await self.process_file(item, session, documents)
                elif item.get('type') == 'dir':
                    await self.process_directory(item.get('url'), session, documents)

        except Exception as e:
            print(f"Error processing directory {url}: {str(e)}")

    async def process_file(self, item: dict, session: aiohttp.ClientSession, documents: List[Document]):
        """Process individual files"""
        try:
            if not all(key in item for key in ['download_url', 'path', 'html_url', 'sha']):
                print(f"Malformed file item: {item}")
                return

            async with session.get(item['download_url']) as file_response:
                if file_response.status != 200:
                    print(f"Failed to download {item['path']}")
                    return

                content = await file_response.text()
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": item['html_url'],
                        "file_path": item['path'],
                        "type": "code",
                        "sha": item['sha']
                    }
                ))
        except Exception as e:
            print(f"Error processing file {item.get('path', 'unknown')}: {str(e)}")

    async def process_pulls(self, repo: str, session: aiohttp.ClientSession) -> List[Document]:
        """Process pull requests"""
        try:
            async with session.get(f'https://api.github.com/repos/{repo}/pulls') as response:
                if response.status != 200:
                    print(f"Error fetching pulls: {await response.text()}")
                    return []

                pulls = await response.json()
                if not isinstance(pulls, list):
                    return []

                return [
                    Document(
                        page_content=f"PR #{pr['number']}: {pr['title']}\n{pr['body']}",
                        metadata={
                            "source": pr['html_url'],
                            "type": "pull_request",
                            "number": pr['number'],
                            "state": pr['state']
                        }
                    )
                    for pr in pulls if isinstance(pr, dict)
                ]
        except Exception as e:
            print(f"Error processing pulls: {str(e)}")
            return []

    async def process_commits(self, repo: str, session: aiohttp.ClientSession) -> List[Document]:
        """Process commits"""
        try:
            async with session.get(f'https://api.github.com/repos/{repo}/commits') as response:
                if response.status != 200:
                    print(f"Error fetching commits: {await response.text()}")
                    return []

                commits = await response.json()
                if not isinstance(commits, list):
                    return []

                return [
                    Document(
                        page_content=f"Commit: {commit['commit']['message']}\nAuthor: {commit['commit']['author']['name']}",
                        metadata={
                            "source": commit['html_url'],
                            "type": "commit",
                            "sha": commit['sha'],
                            "date": commit['commit']['author']['date']
                        }
                    )
                    for commit in commits if isinstance(commit, dict)
                ]
        except Exception as e:
            print(f"Error processing commits: {str(e)}")
            return []

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents using recursive text splitter"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True
        )
        return text_splitter.split_documents(documents)

    def initialize_vector_store(self, chunks: List[Document]):
        """Initialize Qdrant vector store"""
        return Qdrant.from_documents(
            chunks,
            self.embeddings,
            url="http://localhost:6333",
            collection_name=self.repo
        )
    
    def get_all_agents(self):
        """Return all available Qdrant collections"""
        collections = self.qdrant_client.get_collections()
        return [collection.name for collection in collections.collections]

    def create_qa_chain(self, vector_store):
        """Create QA chain with custom prompt"""
        prompt_template = """Analyze the following GitHub repository content to answer the question.
        Consider code files, pull requests, and commits in your response, but do not include the question in your answer. 
        Also, DO NOT speak of the context provided, but instead just use it to answer the question naturally.

        Context:
        {context}

        Question: {question}

        Provide a detailed answer with references to specific files, PRs, or commits.
        IF the query is just a simple hi, hello, yo, what's good, ect, just answer with How can I help you today?
        Answer:"""
        
        QA_PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True
        )

    def init_vector_dbs(self):
        qa_chains = self.qdrant_client.get_collections()
        res = {}
        for chain in qa_chains.collections:
            vector_store = Qdrant(client=self.qdrant_client, collection_name=chain.name, embeddings=self.embeddings)
            res[chain.name] = self.create_qa_chain(vector_store)
        return res