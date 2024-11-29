# Chatbot-with-gmail-func

README
Project: NovaSynth Chatbot with RAG and Email Integration
This project provides a chatbot interface capable of retrieving company details using Retrieval-Augmented Generation (RAG) and sending the company's profile via email upon user request. The chatbot integrates a FastAPI server, Pinecone for vector storage, OpenAI for LLM responses, and SMTP for email delivery.

1. Setup Instructions
Prerequisites
Python: Version 3.8 or later.
Pinecone Account: For vector storage of knowledge base documents.
OpenAI API Key: To use GPT-based models for conversation.
SMTP Email Account: For sending emails.
Steps
Clone Repository:

bash
Copy code
git clone https://github.com/M-A-K-K/Chatbot-with-gmail-func.git
   cd Chatbot-with-gmail-func
Install Dependencies: Install required libraries using pip:

bash
Copy code
pip install -r requirements.txt
Set Up Environment Variables: Create a .env file in the root directory and add:

env
Copy code
OPENAI_API_KEY=your_openai_api_key
SENDER_EMAIL=your_email@example.com
SENDER_PASSWORD=your_email_password
Prepare Data:

Add the company's profile or other knowledge base documents in a structured format.
Store the profile file in the Data/ directory (e.g., Data/company.txt).
Run the Application:

bash
Copy code
uvicorn main:app --host 0.0.0.0 --port 8000
Access the Chatbot:

WebSocket endpoint: ws://localhost:8000/chat
REST endpoint to verify: GET http://localhost:8000/
2. Architecture Overview
Components
FastAPI Server:

Provides REST and WebSocket interfaces for interaction.
Handles user inputs and integrates LLM-based responses.
RAG (Retrieval-Augmented Generation):

Vector Store (Pinecone): Indexes company documents for retrieval.
LLM (OpenAI GPT): Generates responses based on retrieved documents and chat history.
Email Sender:

Uses Python's smtplib to send emails with attachments.
Middleware:

CORS Middleware enables communication from any origin.
Workflow
User Interaction:
User queries are received through a WebSocket.
Processing:
If the query includes an email request, the chatbot sends the requested profile.
If no email is provided, the chatbot prompts the user for one.
Response Generation:
RAG retrieves relevant documents and generates responses using GPT.
Email Delivery:
Company profiles are sent as email attachments if requested.


3. Key Design Decisions


RAG Implementation:

Document Retrieval: Ensures relevant and concise responses by focusing on stored company knowledge.

Context-Aware Chat: Maintains chat history to align future responses with past interactions.

Email Workflow:

Proactive Email Request: If an email is missing, the system explicitly asks for it.

Attachment Validation: Ensures the profile file exists before sending.

Modular Design:

Tools like email and RAG components are designed as independent modules for easy maintenance.

5. Known Limitations

Email Validation:


Only basic email validation using regex is implemented; invalid or unreachable emails won't be caught.

Limited Scalability:


Chat history is stored in memory during the session. Persistent storage (e.g., a database) is needed for scalability.

Single Profile File:


The system currently supports sending a single profile file. Multi-document support can be added.

Error Handling:

Some runtime errors (e.g., email server failure, vector store unavailability) are not fully handled.

5. Future Improvements

Advanced Email Validation:


Add verification mechanisms to check if an email is valid and reachable.

Persistent Chat History:

Store chat history in a database for better tracking and analytics.

Enhanced RAG:

Expand the knowledge base with multi-document indexing and hierarchical retrieval strategies.

User Authentication:

Add user authentication to enhance security and customize responses.

Improved UI:

Build a front-end interface for better interaction and usability.

Asynchronous Email Handling:

Allow email sending in the background to improve responsiveness.
