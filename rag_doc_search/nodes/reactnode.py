"""LangGraph nodes for RAG workflow + ReAct Agent inside generate_content"""

from typing import List, Optional
from rag_doc_search.state.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
#from langgraph.prebuilt import create_react_agent OLD deprecated   
from langchain.agents import create_agent

# Wikipedia tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun


class RAGNodes:
    """Contains the node functions for the RAG workflow and ReAct agent"""

    def __init__(self, retriever, llm):
        """Initializes the RAGNodes with a retriever and an LLM.
        
        Args:
            retriever: The retriever to fetch documents.
            llm: The language model to generate answers.
        """
        self.retriever = retriever
        self.llm = llm
        self._agent = None # lazy initialization of ReAct agent

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Retrieves documents based on the user's question.

        Args:
            state (RAGState): The current state containing the user's question.

        Returns:
            RAGState: Updated state with retrieved documents.
        """
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question, retrieved_docs=docs, answer="")

    def _build_tools(self) -> List[Tool]:
        """Build retriever + wikipedia tools"""

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
            name="retriever",
            description="Fetch passages from indexed corpus.",
            func=retriever_tool_fn,
        )

        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="en")
        )
        wikipedia_tool = Tool(
            name="wikipedia",
            description="Search Wikipedia for general knowledge.",
            func=wiki.run,
        )

        return [retriever_tool, wikipedia_tool]
    
    def _build_agent(self) -> None:
        """Builds the ReAct agent with tools. """
        tools = self._build_tools()
        system_prompt = (
            "You are a helpful RAG agent. "
            "Prefer 'retriever' for user-provided docs; use 'wikipedia' for general knowledge. "
            "Return only the final useful answer."
        )
        self._agent = create_agent(model=self.llm, tools=tools,  system_prompt=system_prompt)

    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate answer using ReAct agent with retriever + wikipedia.
        """
        if self._agent is None:
            self._build_agent()

        result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})

        messages = result.get("messages", [])
        answer: Optional[str] = None
        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", None)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate answer."
        )