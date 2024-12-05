import os
import re
import uuid
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
import smtplib
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from typing import Any
from typing import ClassVar
from pydantic import Field
from langchain.agents import Tool, initialize_agent
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from langchain_core.tools import BaseTool
from langchain.chains import ConversationChain
from email import encoders
from pydantic import PrivateAttr

from langchain.memory import ConversationBufferMemory

# Set up conversation memory
memory = ConversationBufferMemory()

load_dotenv()

app = FastAPI()
chat_history = []
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

def send_email_with_attachment(receiver_email: str, file_path: str):
    """
    Sends an email with a predefined attachment.
    """
    try:
        sender_email = os.getenv("SENDER_EMAIL")
        sender_password = os.getenv("SENDER_PASSWORD")
        
        # Predefined message
        message = """
        Dear Recipient,

        Please find attached the company profile for NovaSynth Tech Solutions.

        Best regards,
        NovaSynth Team
        """
        
        # Create email
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = "NovaSynth Tech Solutions Company Profile"
        msg.attach(MIMEText(message, 'plain'))

        # Attach file
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

        # Send email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return "Email Sent Successfully"
    except Exception as e:
        return f"Error Sending Email: {str(e)}"

# Updated EmailSenderTool
class EmailSenderTool(BaseTool):
    name: str = "Email Sender Tool"
    description: str = """
    Automatically sends the company profile to the provided email address.
    Use this tool when you need to send the company profile to a specific email address.
    If user asks the email address it provided you must return it.
    ** Strictly follow:**
    1. You must get email address from the user.
    Input should include only the recipient's email address as 'receiver_email'.
    """
    file_path: str = Field(..., description="Path to the company profile file")
    inputs: ClassVar[list[str]] = ["receiver_email"]

    def _run(self, inputs: Any):
        try:
            # Handle input being a string
            if isinstance(inputs, str):
                raw_email = inputs
            # Handle input being a dictionary
            elif isinstance(inputs, dict):
                raw_email = inputs.get("receiver_email", "")
            else:
                return "Error: Invalid input type. Expected string or dictionary."

            # Extract and validate the email address
            receiver_email = extract_email(raw_email)
            if not receiver_email:
                return "Error: Invalid or missing email address."

            # Send email
            result = send_email_with_attachment(receiver_email, self.file_path)
            return result
        except Exception as e:
            return f"Error sending email: {str(e)}"

    def _arun(self, inputs: dict):
        raise NotImplementedError("This tool does not support async operations.")



class PineconeQueryTool(BaseTool):
    name: str = "Pinecone Query Tool"
    description: str = """
    Use this tool when a user asks about NovaSynth Tech Solutions and for general greetings. 
    Provide short, concise and accurate answers based on the provided context.
    If user asks do greetings, also respond to them.
    - Provide  exact information what is user is asking.
    - Do not provide xtra information.
    - If the user greets you (e.g., "Hello", "Hi"), respond warmly without invoking any tools.  
    - Example: "Hello! How can I assist you today?"
    For example:- If user asks about the company's mission, provide the mission statement.
    - If user asks about the company's history, provide the history.
    - If user asks about the company's services, provide the services.
    - If user asks about the company's team, provide the team members.
    - If user asks about the company's contact information, provide the contact information.
    - If user asks about the company's location, provide the location.
    - If user asks about the company's website, provide the website.
    - If user asks about the company's social media, provide the social media.
    - If user asks about the company's blog, provide the blog.
    - If user asks about the company's events, provide the events.
    - If user asks about the company's awards, provide the awards.
    - If user asks about the company's partners, provide the partners.
    - If user asks about the company's clients, provide the clients.
    - If user asks about the company's press releases, provide the press releases.
    - If user asks about the company's location, provide the location.
    """

    def _run(self, query: str):
        try:
            # Initialize embedding model
            embedding_model = OpenAIEmbeddings(
                api_key="Uq6en", 
                model="text-embedding-3-small"
            )

            # Initialize vectorstore
            vectorstore = PineconeVectorStore(index_name="company-bot", embedding=embedding_model)

            # Retrieve relevant documents
            retriever = vectorstore.as_retriever(top_k=3)
            retrieved_docs = retriever.get_relevant_documents(query)

            # Extract context from retrieved documents
            context = "\n".join([doc.page_content for doc in retrieved_docs])

            # Initialize LLM
            llm = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.1,
                openai_api_key="6en"
            )

            # Define QA Prompt Template
            qa_prompt = ChatPromptTemplate.from_template(
                """You are an intelligent assistant specializing in providing accurate and concise information about NovaSynth Tech Solutions.
                
                **Instructions:**  
                - Always provide information that user wants.
                - Do not provide extra information.
                - Always respond based on the provided context below.  
                - Provide clear, actionable answers in no more than 3 sentences.  
                - If the user expresses interest, offer to send the company profile via email.  
                - If the user greets you (e.g., "Hello", "Hi"), respond warmly without invoking any tools.  
                  - Example: "Hello! How can I assist you today?"
                **Context:**  
                {context}  

                **Rules:**  
                1. Only respond using information from the context.  
                2. If no relevant information is found, inform the user politely and offer to follow up via email if necessary.  
                3. Avoid providing speculative or unrelated answers.  

                User Input: {input}  
                """
            )

            # Create a chain to handle QA
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(vectorstore.as_retriever(top_k=3), question_answer_chain)

            ai_respo = rag_chain.invoke({
                "input": query,
                "context": context,
                "chat_history": chat_history
            })
            response_text = ai_respo["answer"]
            return response_text

        except Exception as e:
            return f"Error querying Pinecone: {str(e)}"

    def _arun(self, query: str):
        # This method remains unimplemented for synchronous-only usage
        raise NotImplementedError("This tool does not support async operations.")



# WebSocket route with Pinecone tool
@app.websocket("/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()

    # Create LLM and tools
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=1)
    email_tool = EmailSenderTool(file_path="Data/company.txt")

    # Use PineconeQueryTool with hardcoded API key and index name
    pinecone_tool = PineconeQueryTool()

    tools = [email_tool, pinecone_tool]

    # Define the agent prompt
    prompt_template = """
    You are an intelligent and conversational assistant for NovaSynth Tech Solutions.  
    You should only invoke tools when strictly required. For all other inputs, respond naturally without invoking tools.  

    **Rules for Behavior:**
    1. **Greeting Handling**:  
    - If the user greets you (e.g., "Hello", "Hi"), respond warmly without invoking any tools.  
    - Example: "Hello! How can I assist you today?"
    2. **General Queries**:  
    - Answer general inquiries directly without tools if the answer is simple or self-contained.  
    - Example: If the user asks, "What does NovaSynth do?", respond with a brief explanation:  
        "NovaSynth Tech Solutions specializes in innovative technology solutions and services."
    3. **Tool Usage**:  
    - Use tools only when strictly required to retrieve or process specific information.  
    - Example: Sending the company profile or retrieving data from Pinecone.
    4. **Fallback Handling**:  
    - If you encounter errors (e.g., invalid tool format), respond directly to the user's query without retrying the tool.  
    - Example: "I'm sorry, I couldn't process that. Can I assist you with something else?"

    **Important Notes:**  
    - Do not invoke tools for greetings or casual conversations.  
    - Avoid retrying failed tool actions; respond naturally instead.  

    **Examples of Proper Behavior**:
    - User: "Hi"  
    Assistant: "Hello! How can I assist you today?"
    - User: "Send me the company profile."  
    Assistant: "Sure! Please share your email address, and I'll send it to you."  
    (Uses the EmailSenderTool only after email address is provided.)
    - User: "Tell me about NovaSynth Tech Solutions."  
    Assistant: "NovaSynth specializes in providing innovative technology solutions to businesses worldwide."

    Chat History: {chat_history}  
    User Input: {input}
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["tools","chat_history", "input"])

    # Initialize the agent with session-specific memory
    agent = initialize_agent(
        agent_type="chat-conversational-react-description",
        tools=tools,
        llm=llm,
        verbose=True,
        memory=None,  # Define memory if needed
        prompt=PROMPT,
        handle_parsing_errors=True 
    )

    try:
        while True:
            # Receive user input
            user_message = await websocket.receive_text()

            if not user_message:
                await websocket.send_text("Error: No message received. Please send a message.")
                continue

            # Add user message to the chat history
            chat_history.append({"role": "user", "message": user_message})

            # Get the agent's response (it will automatically choose the correct tool based on the input)
            try:
                response = agent.run(user_message)

                # Add assistant's response to the chat history
                chat_history.append({"role": "assistant", "message": response})

                # Send the response back to the user
                await websocket.send_text(response)

            except Exception as e:
                error_message = f"Error during processing: {str(e)}"
                await websocket.send_text(error_message)

    except WebSocketDisconnect:
        print("Client disconnected")
        session_id = str(uuid.uuid4())
        history_folder = "history"

        if not os.path.exists(history_folder):
            os.makedirs(history_folder)

        # Save session memory history to a file for persistence
        with open(f"{history_folder}/chat_history_{session_id}.json", "w") as history_file:
            json.dump(chat_history, history_file, indent=4)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)








