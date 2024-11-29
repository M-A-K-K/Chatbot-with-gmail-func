NovaSynth Chatbot with RAG and Email Integration
Welcome to the NovaSynth Chatbot project! This intelligent chatbot is designed to retrieve company information using Retrieval-Augmented Generation (RAG) and send the company profile via email upon request. Built with cutting-edge technologies like FastAPI, Pinecone, and OpenAI, it provides a seamless and efficient interaction experience.

🚀 Features
Smart Retrieval: Uses RAG to fetch relevant company details from a knowledge base.
Email Integration: Sends the company profile as an email attachment upon user request.
Real-Time Interaction: Communicate via WebSocket for instant responses.
Scalable & Modular: Designed with independent components for easy maintenance and future enhancements.
📦 Setup Instructions
Prerequisites
Ensure the following tools and accounts are ready:

Python: Version 3.8 or later.
Pinecone Account: For vector storage of knowledge base documents.
OpenAI API Key: To leverage GPT-based language models.
SMTP Email Account: Gmail or similar account for email delivery.
Steps to Run
1. Clone the Repository
bash
Copy code
git clone https://github.com/M-A-K-K/Chatbot-with-gmail-func.git
cd Chatbot-with-gmail-func
2. Install Dependencies
bash
Copy code
pip install -r requirements.txt
3. Configure Environment Variables
Create a .env file in the root directory with the following:

env
Copy code
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_environment
SENDER_EMAIL=your_email@example.com
SENDER_PASSWORD=your_email_password
4. Prepare Data
Add the company profile or other knowledge base documents in the Data/ directory.
Example: Save the company profile as Data/company.txt.
5. Run the Application
bash
Copy code
uvicorn main:app --host 0.0.0.0 --port 8000
6. Access the Chatbot
WebSocket Endpoint: ws://localhost:8000/chat
REST Endpoint (for testing): GET http://localhost:8000/
🛠 Architecture Overview
Core Components
FastAPI Server

Handles REST and WebSocket communication.
Integrates user input processing and response generation.
Retrieval-Augmented Generation (RAG)

Vector Store (Pinecone): Indexes and retrieves documents.
LLM (OpenAI GPT): Generates context-aware and relevant responses.
Email Sending Module

Uses Python's smtplib for email functionality.
Sends company profiles as email attachments.
Middleware

CORS Middleware: Enables cross-origin requests.
Workflow
User Interaction
Receives input via WebSocket.
Processing
Checks for email requests and retrieves relevant documents.
Prompts users for email if not provided.
Response Generation
Uses RAG to fetch relevant details and generate a GPT-based response.
Email Delivery
Sends company profiles as attachments if requested.
📌 Key Design Decisions
RAG Implementation

Focused Retrieval: Ensures concise responses by retrieving contextually relevant knowledge.
Chat Continuity: Maintains chat history for better contextual responses.
Email Workflow

Proactive Requests: Explicitly prompts users for an email if missing.
Validation: Ensures file attachment exists before sending.
Modular Components

Email sending and RAG processes are designed independently for better scalability.
⚠ Known Limitations
Basic Email Validation

Uses regex for validation; unreachable or invalid emails are not flagged.
Limited Scalability

Chat history is stored in memory during a session; persistent storage like a database is needed for scaling.
Single File Support

Currently supports a single profile file for email attachment.
Error Handling

Limited handling of runtime issues (e.g., email server or vector store outages).
🌟 Future Improvements
Enhanced Email Validation

Add mechanisms to verify email validity and deliverability.
Persistent Chat History

Use a database to store and analyze user interactions.
Improved RAG

Expand to support multi-document indexing and hierarchical retrieval.
User Authentication

Add login and authentication for secure and personalized responses.
User-Friendly Interface

Build a front-end for better interaction.
Asynchronous Email Handling

Implement background email-sending for improved responsiveness.
📖 About Retrieval-Augmented Generation (RAG)
RAG enhances chatbot responses by combining document retrieval with generative AI, ensuring contextually relevant and accurate information.

How It Works
Document Embedding: Knowledge base files are converted to vector embeddings using OpenAI models.
Vector Store: Embeddings are stored in Pinecone for similarity search.
Query Matching: User input is matched against stored vectors to retrieve top documents.
Response Generation: GPT generates a cohesive response using the retrieved context.
💡 Get Started Today!
Clone the repository, follow the setup instructions, and experience the power of intelligent, RAG-enabled chat with email integration.

