

# import os
# from langchain_openai import OpenAIEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.schema import Document
# from pinecone import Pinecone as PineconeClient, ServerlessSpec
# from langchain_pinecone import PineconeVectorStore
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# # Initialize Pinecone client
# pc = PineconeClient(api_key=PINECONE_API_KEY, environment="us-east-1")
# index_name = "company-bot"

# # Ensure the Pinecone index exists
# try:
#     if index_name not in pc.list_indexes().names():
#         pc.create_index(
#             name=index_name,
#             dimension=1536,
#             metric="cosine",
#             spec=ServerlessSpec(
#                 cloud="aws",
#                 region="us-east-1"
#             )
#         )
#     print(f"Index '{index_name}' created or exists.")
# except Exception as e:
#     print(f"Error creating index: {e}")

# # Initialize embedding model
# embedding_model = "text-embedding-3-small"  
# embedd = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=embedding_model)
# vectorstore = PineconeVectorStore(index_name=index_name, embedding=embedd)

# # Function to process and embed text file content
# def process_and_embed_text(file_path):
#     """
#     Process and embed data from a text file into Pinecone.

#     Args:
#         file_path (str): Path to the text file.
#     """
#     # Read the text file
#     try:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             content = file.read()
#     except Exception as e:
#         print(f"Error reading file: {e}")
#         return

#     print(f"Loaded content from '{file_path}'. Starting processing...\n")

#     # Convert the text into Document format
#     document = Document(page_content=content, id="company-data")
    
#     # Split into smaller chunks
#     text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10)
#     chunks = text_splitter.split_documents([document])

#     # Embed and store the chunks
#     embedding_vectors = embedd.embed_documents([doc.page_content for doc in chunks])
#     vectorstore.add_documents(documents=chunks, embeddings=embedding_vectors)

#     print(f"Successfully embedded and stored {len(chunks)} chunks.")

# # Path to the company text file
# company_file_path = "./Data/company.txt"

# # Process and embed the text file
# process_and_embed_text(company_file_path)

import os
import re
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_email(message: str):
    email_regex = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    match = re.search(email_regex, message)
    if match:
        return match.group(0)
    return None

def send_email_with_attachment(receiver_email: str, message: str, file_path: str):
    try:
        sender_email = os.getenv("SENDER_EMAIL")
        sender_password = os.getenv("SENDER_PASSWORD")
        msg = MIMEMultipart()
        msg['From'] = "m.abdulkabirkhan@gmail.com"
        msg['To'] = receiver_email
        msg['Subject'] = "NovaSynth Tech Solutions Company Profile"

        msg.attach(MIMEText(message, 'plain'))

        if os.path.exists(file_path):
            with open(file_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename={os.path.basename(file_path)}'
                )
                msg.attach(part)
        else:
            return f"Error: File not found at {file_path}"

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return "Email Sent Successfully"
    except Exception as e:
        return f"Error Sending Email: {str(e)}"

index_name = "company-bot"
embedding_model = 'text-embedding-3-small'
embedd = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    model=embedding_model
)

@app.post("/")
async def home():
    return {"success": True, "message": "Chat endpoint is working!"}

@app.websocket("/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    chat_history = []

    try:
        while True:
            user_message = await websocket.receive_text()

            if not user_message:
                await websocket.send_text("Error: No message received. Please send a message.")
                continue

            chat_history.append({"role": "user", "message": user_message})

            email_address = extract_email(user_message)
            if email_address:
                file_path = os.path.join("Data", "company.txt")
                email_message = "Please find attached the NovaSynth Tech Solutions company profile."
                email_status = send_email_with_attachment(email_address, email_message, file_path)

                chat_history.append({"role": "system", "message": email_status})

                await websocket.send_text(email_status)
                continue

            try:
                llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
                vectorstore = PineconeVectorStore(index_name=index_name, embedding=embedd)

                qa_prompt = ChatPromptTemplate.from_template(
                    """You are a knowledgeable assistant providing accurate information about NovaSynth Tech Solutions. 
                    Always respond based on the provided context.

                    **Instructions to follow:**
                    1. Always Provide first a brief company overview in no more than 2 sentences.
                    2. For detailed inquiries, retrieve relevant information from the knowledge base and present it clearly.
                    3. Offer to email the company profile to the user if they express interest. 
                       - If the user agrees, ask for their email address.
                       - Send the profile to the provided email.

                    **Rules:**  
                    - Refrain from answering out of the scope of the provided context.
                    - Keep concise, and user-friendly.
                    History: {chat_history}
                    System Context: {context}
                    User Input: {input}"""
                )

                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(vectorstore.as_retriever(top_k=3), question_answer_chain)

                ai_respo = rag_chain.invoke({
                    "input": user_message,
                    "context": qa_prompt,
                    "chat_history": chat_history
                })
                response_text = ai_respo["answer"]

                chat_history.append({"role": "assistant", "message": response_text})

                await websocket.send_text(response_text)

            except Exception as e:
                error_message = f"Error during processing: {str(e)}"
                chat_history.append({"role": "system", "message": error_message})
                await websocket.send_text(error_message)

    except WebSocketDisconnect:
        print("Client disconnected")
        session_id = str(uuid.uuid4())
        history_folder = "history"

        if not os.path.exists(history_folder):
            os.makedirs(history_folder)

        with open(f"{history_folder}/chat_history_{session_id}.json", "w") as history_file:
            import json
            json.dump(chat_history, history_file, indent=4)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
