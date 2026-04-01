import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#changed: use Google Generative AI for embeddings instead of Open Ai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
import asyncio


try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


#load environment
load_dotenv()
#changed: use google api key instead of open ai
#ensure your .env file has GOOGLE_API_KEY = 'your_google_api_key'
google_api_key = os.getenv("GOOGLE_API_KEY")

#Paths(keeping as is)
DOCS_FOLDER = "doc/"
VECTOR_DB_FOLDER = "vector_store/"

#create vector store folder if it doesn't exist
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

#collect all PDf files (keeping as is)
pdf_files = [os.path.join(DOCS_FOLDER, f) for f in os.listdir(DOCS_FOLDER) if f.endswith('.pdf')]

#load documents (keeping as is, with corrected print statement)
all_docs = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()
    all_docs.extend(docs)
    #corrected print statement to show the file being processed
    print(f"Loaded {len(docs)} documents from {os.path.basename(pdf_file)}")

#split documents into chunks (keeping as is)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)     #learn about this it is imp 

docs_split = splitter.split_documents(all_docs)
print(f"Total number of chunks created: {len(docs_split)}")

#create embedding
#changed = initialize GoogleGenerativeAIEmbeddings with the API key
#the model "embedding-001" is typically used for Google Generative AI embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="embedding-001", api_key=google_api_key)
print("Google Generative AI embeddings initialized.")


#create FAISS index (keeping as is)
db = FAISS.from_documents(docs_split, embeddings)
print("FAISS index created.")

#save index to disk (keeping as is, with corrected print statement)
print(f"Vector Db saved to {VECTOR_DB_FOLDER}")


db.save_local(VECTOR_DB_FOLDER)
