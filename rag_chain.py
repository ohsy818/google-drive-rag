import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from vector_store import VectorStoreManager

class RAGChain:
    def __init__(self):
        print("Initializing RAG Chain...")
        load_dotenv()
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Initialize ChatOpenAI with debugging
        print("Initializing ChatOpenAI...")
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            api_key=openai_api_key
        )
        print("ChatOpenAI initialized successfully")
        
        self.vector_store = VectorStoreManager()
        
        # Create custom prompt template
        template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        
        Answer: Let me help you with that.
        """
        
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        print("Creating RetrievalQA chain...")
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.get_retriever(),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": self.prompt
            }
        )
        print("RetrievalQA chain created successfully")
    
    def answer_question(self, question: str, filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Answer a question using the RAG chain."""
        try:
            print(f"\nProcessing question: {question}")
            if filter:
                print(f"Using filter: {filter}")

            # First try direct similarity search
            print("Performing similarity search...")
            docs = self.vector_store.similarity_search(question, filter=filter)
            print(f"Found {len(docs)} relevant documents")
            
            if not docs:
                return {
                    "answer": "I don't have enough information to answer that question.",
                    "sources": []
                }

            # Run the chain
            print("Running RetrievalQA chain...")
            result = self.chain({"query": question})
            
            # Extract answer and sources
            answer = result.get("result", "I don't have enough information to answer that question.")
            source_documents = result.get("source_documents", [])
            
            print(f"Answer: {answer}")
            print(f"Number of source documents: {len(source_documents)}")
            
            return {
                "answer": answer,
                "sources": [doc.page_content for doc in source_documents]
            }
        except Exception as e:
            print(f"Error in answer_question: {e}")
            return {
                "answer": "An error occurred while processing your question.",
                "sources": []
            }
    
    def process_and_store_documents(self, documents: list[Document]) -> bool:
        """Process and store documents in the vector store."""
        try:
            print(f"\nProcessing {len(documents)} documents...")
            return self.vector_store.add_documents(documents)
        except Exception as e:
            print(f"Error in process_and_store_documents: {e}")
            return False

# Example usage
if __name__ == "__main__":
    rag = RAGChain()
    result = rag.answer_question(
        "What is the budget for Project A?",
        filter={"storage_type": "Local"}
    )
    print(f"Answer: {result['answer']}")
    print("\nSources:")
    for source in result["sources"]:
        print(f"- {source[:100]}...") 