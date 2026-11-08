from rag_doc_search.state.rag_state import RAGState
class RAGNodes:
    """
    Defines the nodes for the RAG pipeline.

    The nodes consist of two main functions:
        1. retrieve_docs: Retrieves documents based on the user's question.
        2. generate_answer: Generates an answer based on the retrieved documents.
    """

    def __init__(self, retriever, llm):
        """Initializes the RAGNodes with a retriever and an LLM.
        
        Args:
            retriever: The retriever to fetch documents.
            llm: The language model to generate answers.
        """
        self.retriever = retriever
        self.llm = llm

    def retrieve_docs(self, state: RAGState)-> RAGState:
        """Retrieves documents based on the user's question.

        Args:
            state (RAGState): The current state containing the user's question.

        Returns:
            RAGState: Updated state with retrieved documents.
        """
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question, retrieved_docs=docs, answer="")

    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate answer from retrieved documents node
        
        Args:
            state: Current RAG state with retrieved documents
            
        Returns:
            Updated RAG state with generated answer
        """
        # Combine retrieved documents into context
        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
        
        # Create prompt
        prompt = f"""Answer the question based on the context.

        Context:
        {context}

        Question: {state.question}"""
        
        # Generate response
        response = self.llm.invoke(prompt)
        
        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=response.content
        )