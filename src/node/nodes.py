"""LangGraph Nodes for RAG workflow"""

from src.state.rag_state import RAGState

class RAGNodes:
    """ Contains node functions for the RAG workflow"""

    def __init__(self, retriever, llm):
        """ Initialize the RAG nodes
        
        Arguments:
            retriever: The document retriever instance
            llm: The large language model instance"""
        
        self.retriever = retriever
        self.llm = llm

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """ Retrieves relevant document nodes
        
        Arguments:
            state: Current RAG state
            
        Returns:
            Updated RAG state with the retrieved documents"""
        
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question = state.question,
            retrieved_docs = docs
        )
    
    def generate_answer(self, state:RAGState) -> RAGState:
        """ Generate answer from retrieved documents node.
        
        Arguments:
            state: The current RAG state with retrieved documents.
            
        Returns:
            Updated RAG state witht the genrated answer."""
        
        # Combine retrieved documents into context
        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])

        # Create a prompt
        prompt = f"""Answer the question based on the context
        Context:{context}
        Question: {state.quesion}"""

        #Generate a response
        response = self.llm.invoke(prompt)

        return RAGState(
            question = state.question,
            retrieved_docs  = state.retrieved_docs,
            answer = response.content
        )
