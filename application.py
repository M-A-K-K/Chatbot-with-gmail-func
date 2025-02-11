
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
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate
import smtplib
from langchain_core.prompts import PromptTemplate
from typing import Any
from typing import ClassVar
from pydantic import Field
from langchain.agents import create_react_agent
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from langchain_core.tools import BaseTool
from email import encoders
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_retrieval_chain

memory = ConversationBufferMemory()

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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

class EmailSenderTool(BaseTool):
    name: str = "EmailSenderTool"
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
    name: str = "PineconeQueryTool"
    description: str = """
     Use this tool when a user asks about NovaSynth Tech Solutions or any information related to its data, products, services, achievements, leadership, or other details provided in the context. 
    - Provide precise, short and concise responses directly addressing the user's query. 
    - Avoid adding any extra or unrelated information.
    - Ensure that all answers are strictly based on the given company data.
    - Invalid or incomplete responses will be penalized.
    """

    def _run(self, query: str):
        try:
            embedding_model = OpenAIEmbeddings(
                api_key=OPENAI_API_KEY,
                model="text-embedding-3-small"
            )
            vectorstore = PineconeVectorStore(index_name="company-bot", embedding=embedding_model)
    
            llm = ChatOpenAI(
                model_name="gpt-4o",
                temperature=0.1, 
                openai_api_key=OPENAI_API_KEY
            )
            qa_prompt = ChatPromptTemplate.from_template(
    """
    You are an intelligent assistant providing concise and well-formatted information about NovaSynth Tech Solutions.
    - Provide answers in short bullet points.
    - If user asks something out of context, politely inform them that you can only provide information based on the provided data.
    - Focus on short answers and avoid unnecessary details.
    - Format the response for readability, using clear headers and subheaders.
    Context: {context}
    User Query: {input}
    """
)

            chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(vectorstore.as_retriever(top_k=3), chain)
            ai_respo = rag_chain.invoke({
                    "input": query,
                    "context": qa_prompt,
                    "chat_history": chat_history
                })
            response_text = ai_respo["answer"]
            return response_text
        except Exception as e:
            return f"Error querying Pinecone: {str(e)}"

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async operations.")


@app.websocket("/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5)
    email_tool = EmailSenderTool(file_path="Data/company.txt")
    Pinecone_tool = PineconeQueryTool()
    tools = [email_tool, Pinecone_tool]
    tool_names = [tool.name for tool in tools]

    print("Available Tool Names:", tool_names)
    prompt_template = """
You are an intelligent assistant with access to the following tools:
{tools}
- Strictly always respond in a short manner.
-If the input does not requires a tool, do directly.
When a user provides input:
- Analyze the input to determine if it requires a tool (e.g., PineconeQueryTool or EmailSenderTool).
-If the input does not requires a tool, deny the request.
- If the user requests information about the company, retrieve the data using PineconeQueryTool. 
- If this is the first time the user is asking about the company, after providing the answer, ask if they would like more detailed information sent via email.
- If the user agrees, collect their email address and use EmailSenderTool to send the company profile.
- If the user provides an email address directly, send the detailed company profile immediately using EmailSenderTool.

**Strictly follow this reasoning process**:
Thought: Describe your thought process.
Action: Specify the tool you are using (e.g., PineconeQueryTool or EmailSenderTool).
Action Input: Provide the necessary input for the tool.
Observation: Record the tool's response.
(Repeat Thought/Action/Action Input/Observation as needed.)
Final Answer: Provide the final response to the user, ensuring it aligns with the interaction rules above.

Chat History: {chat_history}
User Input: {input} 
Tool Names: {tool_names}
Tools: {tools}
Agent Scratchpad: {agent_scratchpad}
"""

# When a user provides input:
# - Analyze the input to determine if it requires a tool (PineconeQueryTool or EmailSenderTool).
# - If the user requests information about the company, retrieve the data using PineconeQueryTool. 
# - If this is the first time the user is asking about the company, after providing the answer, ask if they would like more detailed information sent via email.
# - If the user agrees, collect their email address and use EmailSenderTool to send the company profile.
# - If the user provides an email address directly, send the detailed company profile immediately using EmailSenderTool.

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["tools", "tool_names", "chat_history", "input", "agent_scratchpad"]
    )

    agent = create_react_agent(
        tools=[email_tool, Pinecone_tool],
        llm=llm,
        prompt=PROMPT
    )
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=[email_tool, Pinecone_tool],
        verbose=True,
        handle_parsing_errors=True,
        max_iterations= 10,
    )

    chat_history = []  
    try:
        while True:
            user_message = await websocket.receive_text()

            if not user_message:
                await websocket.send_text("Error: No message received. Please send a message.")
                continue

            chat_history.append({"role": "user", "message": user_message})

            try:
                input_data = {
                    "input": user_message,
                    "chat_history": json.dumps(chat_history),
                    "tools": [tool.__class__.__name__ for tool in tools],
                    "tool_names": [tool.name for tool in tools], 
                    "agent_scratchpad": "",
                }


                response = agent_executor.invoke(input_data)

                if isinstance(response, dict):  
                    output_value = response.get("output", "No output found")
                else:
                    output_value = "Invalid response format"

                chat_history.append({"role": "assistant", "message": output_value})

                await websocket.send_text(output_value)

            except Exception as e:
                error_message = f"Error during processing: {str(e)}"
                await websocket.send_text(error_message)

    except WebSocketDisconnect:
        print("Client disconnected")
        session_id = str(uuid.uuid4())
        history_folder = "history"

        if not os.path.exists(history_folder):
            os.makedirs(history_folder)

        with open(f"{history_folder}/chat_history_{session_id}.json", "w") as history_file:
            json.dump(chat_history, history_file, indent=4)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
