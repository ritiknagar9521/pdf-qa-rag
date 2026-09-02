from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import InMemoryVectorStore

## Load the PDF and convert its pages into LangChain documents.
loader = PyPDFLoader("./sample.pdf")
docs = loader.load()
#print(len(docs))

##Split the document into chunks
# Large documents are split into smaller overlapping chunks
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

##user query

# Temporary hard-coded question.
# Later, this will come from the Streamlit interfac
query = "what is the name of student?"

## Retrieve relevant documents
# Search the vector store for the most relevant document chunks.
documents = vector_db.similarity_search(query=query, k=1)

# Combine the retrieved chunks into a single context.
context = " "
for doc in documents:
    context = context + doc.page_content + "\n\n"


# Tell the LLM to answer using the retrieved PDF context.
prompt = """ You are a helpful assistant and provide anser based on the provided context: {context}, question: {query}"""
formatted_prompt = prompt.format(context = context, query = query)


# Generate the final answer
# Send the retrieved context and question to Gemini.
llm = ChatGoogleGenerativeAI(model= "gemini-3.7-flash") 
answer = llm.invoke(formatted_prompt)

print(answer.content)


