"""LangGraph nodes for RAG workflow + ReAct Agent"""

from typing import List, Optional, Dict, Any
from src.state.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import SearxSearchWrapper

# Import for schema
from pydantic import BaseModel, Field

# Constants or settings for search tool
search_wrapper_global = SearxSearchWrapper(searx_host="http://127.0.0.1:8888", k=5)

class WikiInput(BaseModel):
    query: str = Field(..., description="Search query string for the Wikipedia tool")

class RAGNodes:
    """ Contains the node functions for the RAG workflow + ReAct Agent"""

    def __init__(self, retriever, llm):
        self.retriever = retriever 
        self.llm = llm
        self._agent = None

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Classic retriever node"""
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question = state.question,
            retrieved_docs = docs
        )
    
    def _build_tools(self) -> List[Tool]:
        """Build retriever + wikipedia + search tools"""

        def retriever_tool_fn(query: str) -> str:
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found."
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n\n".join(merged)
        
        retriever_tool = Tool(
            name = "retriever",
            description = "Fetch passages from indexed vectorstore: takes one parameter 'query' as the string to search in the vector store.",
            func = retriever_tool_fn
        )

        # Wikipedia tool with explicit schema
        wiki_wrapper = WikipediaAPIWrapper(top_k_results=4, lang="en")
        wiki_query_tool = WikipediaQueryRun(
            api_wrapper=wiki_wrapper,
            args_schema=WikiInput,
            name="wikipedia",
            description="Search Wikipedia for general knowledge; expects a dict with field 'query'"
        )
        def wikipedia_tool_fn(params: Dict[str, Any]) -> str:
            # expects params to be {"query": "..."}
            # Validate schema (this is done by wiki_query_tool.invoke internally)
            return wiki_query_tool.invoke(params)

        wikipedia_tool = Tool(
            name = "wikipedia",
            description = "Search Wikipedia for general knowledge.",
            func = wikipedia_tool_fn
        )

        # Search tool remains similar
        search_tool = Tool(
            name = "search",
            description = "Search the web for current or trending queries; expects parameter 'query' as string.",
            func = search_wrapper_global.run
        )

        return [retriever_tool, wikipedia_tool, search_tool]

    def _build_agent(self):
        """React Agent with tools"""
        tools = self._build_tools()
        system_prompt = (
            "You are a cheerful, helpful, and detail-oriented RAG agent. "
            "Use 'retriever' for documents from the vector store (parameter: query). "
            "Use 'wikipedia' for general knowledge (parameter: query). "
            "Use 'search' for current or trending event search (parameter: query). "
            "Always return only the final useful answer; do not include internal tool workings in your output."
        )

        # If the LLM supports binding tools/schema, ensure to bind tools
        try:
            # Some LLMs (like Gemini / Google GenAI) support bind_tools
            self.llm = self.llm.bind_tools([tool for tool in tools])
        except Exception:
            # If bind_tools not supported, just proceed
            pass

        self._agent = create_react_agent(self.llm, tools=tools, prompt=system_prompt)

    def generate_answer(self, state: RAGState) -> RAGState:
        """ Generate answer using ReAct agent with retriever, wikipedia, search tools"""
        if self._agent is None:
            self._build_agent()

        result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})

        messages = result.get("messages", [])
        answer: Optional[str] = None
        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", None)

        return RAGState(
            question = state.question,
            retrieved_docs = state.retrieved_docs,
            answer = answer or "Could not generate an answer."
        )
