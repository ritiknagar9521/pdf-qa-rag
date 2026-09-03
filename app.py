from dotenv import load_dotenv 
 
load_dotenv() 
 
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI 
from langchain_community.vectorstores import InMemoryVectorStore 
from time import sleep 
import streamlit as st 
 
# Send the retrieved context and question to Gemini. 
llm = ChatGoogleGenerativeAI(model= "gemini-3.7-flash")  
 
# Store the vector database in session state so it is available
# across Streamlit reruns.
if "vector_db" not in st.session_state: 
    st.session_state.vector_db = None 
 
def document_process(path): 
    ## Load the PDF and convert its pages into LangChain documents. 
    loader = PyPDFLoader(path) 
    docs = loader.load() 
     
 
    ##Split the document into chunks 
    #Large documents are split into smaller overlapping chunks 
    # so that relevant information can be retrieved more accurately. 
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200) 
    docs = splitter.split_documents(docs) 
    #print(len(docs)) 
 
 
    ##Create embeddings and vector store 
    # Convert each document chunk into a numerical vector 
    # using Google's Gemini embedding model. 
    embeddings = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-2-preview") 
 
    # Store the document chunks and their embeddings in memory. 
    # This allows us to search for chunks that are semantically 
    # similar to the user's question. 
    vector_db = InMemoryVectorStore.from_documents(documents = docs, embedding = embeddings) 
 
    # Save the vector database so it can be used later by the chat UI.
    st.session_state.vector_db = vector_db 
    st.session_state.document_uploaded = True 
 
 
 
 
st.subheader("Q&A ChatBot") 

# Keep track of whether a document has been uploaded and processed.
if "document_uploaded" not in st.session_state:      
   st.session_state.document_uploaded = False 

## document upload 
 
# Show the uploader only until a document has been processed.
if not st.session_state.document_uploaded: 
    file = st.file_uploader(label="Select Your PDF File", type="pdf") 
    if file: 
        with open("sample.pdf","wb") as f: 
            f.write(file.getvalue()) 

        # Process the uploaded PDF and create its vector database.
        with st.spinner("Understanding..."): 
            document_process("./sample.pdf")    
 
        st.markdown("Document Processed Successfully...")   
        sleep(2) 

        # Rerun the app so the updated session state is reflected
        # and the document-upload section is skipped.
        st.rerun() 

## chat ui 

