
# **🌐 NovaSynth Chatbot with RAG and Email Integration**

NovaSynth Chatbot is an intelligent system that combines **Retrieval-Augmented Generation (RAG)** and email functionality to provide seamless access to company details. With advanced technologies like **FastAPI**, **Pinecone**, and **OpenAI**, this chatbot delivers instant responses and can send the company's profile via email upon request.

---

## **📋 Features**

- 🔍 **Smart Document Retrieval**: Fetches precise information using RAG.
- ✉️ **Email Functionality**: Sends company profiles as email attachments on demand.
- ⏱ **Real-Time Communication**: WebSocket-based instant interaction.
- ⚙️ **Modular and Scalable Design**: Independent components ensure easy maintenance.

---

## **🚀 Quick Setup Guide**

### **Prerequisites**

Make sure you have the following ready:
- **Python**: Version 3.8 or higher.
- **Pinecone Account**: For vector database.
- **OpenAI API Key**: For LLM capabilities.
- **SMTP Email Account**: For sending emails (e.g., Gmail).

---

### **Step-by-Step Instructions**

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/M-A-K-K/Chatbot-with-gmail-func.git
   cd Chatbot-with-gmail-func
   ```

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**  
   Create a `.env` file in the root directory with the following content:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENV=your_pinecone_environment
   SENDER_EMAIL=your_email@example.com
   SENDER_PASSWORD=your_email_password
   ```

4. **Prepare Data**  
   - Add the company profile or knowledge base documents to the `Data/` directory.
   - Example: Save the profile as `Data/company.txt`.

5. **Run the Application**  
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

6. **Access the Chatbot**  
   - **WebSocket**: `ws://localhost:8000/chat`  
   - **REST Endpoint**: `http://localhost:8000/` (for testing).

---

## **📐 Architecture Overview**

### **Core Components**

| Component             | Functionality                                                                 |
|------------------------|------------------------------------------------------------------------------|
| **FastAPI Server**     | Manages WebSocket and REST communication.                                   |
| **RAG System**         | Combines document retrieval and LLM response generation.                   |
| **Email Module**       | Sends company profiles as email attachments using SMTP.                    |
| **Middleware**         | CORS middleware enables cross-origin requests.                             |

---

### **System Workflow**

1. **User Interaction**: Input queries via WebSocket.
2. **Processing**:
   - For email requests: Prompts user for email if not provided.
   - For other queries: Retrieves relevant details from the knowledge base.
3. **Response Generation**: Uses RAG for context-aware, concise responses.
4. **Email Delivery**: Sends the company profile as an email attachment when requested.

---

## **🧩 Key Design Highlights**

- **RAG Implementation**:
  - Retrieves top documents from Pinecone based on user queries.
  - Uses OpenAI's GPT to generate context-aware responses.

- **Email Integration**:
  - Validates the presence of email and file attachment before sending.
  - Prompts user explicitly if an email address is missing.

- **Modular Design**:
  - Independent components for easy scaling and debugging.

---

## **⚠ Known Limitations**

- **Basic Email Validation**:  
  - Uses regex to validate email format but doesn’t verify email deliverability.

- **Limited Scalability**:  
  - Chat history is stored in memory; requires a database for larger-scale use.

- **Single File Support**:  
  - Currently limited to a single profile file for email attachment.

- **Error Handling**:  
  - Limited handling for issues like server or email delivery failures.

---

## **✨ Future Improvements**

- **Advanced Email Validation**:  
  - Add mechanisms to verify email deliverability.

- **Persistent Chat History**:  
  - Use a database to store user interactions for analytics and better UX.

- **Expanded Knowledge Base**:  
  - Support for multi-document indexing and advanced retrieval.

- **User Authentication**:  
  - Add login systems for secure and personalized responses.

- **Improved Interface**:  
  - Develop a front-end for intuitive user interaction.

- **Asynchronous Email Handling**:  
  - Enable background email sending for better responsiveness.

---

## **🔬 How RAG Works**

**Retrieval-Augmented Generation (RAG)** combines document retrieval with generative AI to deliver precise, context-driven responses.  

1. **Document Embedding**: Transforms knowledge base into vector embeddings.  
2. **Vector Search**: Finds the most relevant documents using Pinecone.  
3. **Generative Response**: GPT generates a response based on the retrieved context.

---

## **📞 Support**

For any issues or inquiries, please reach out via email: [m.abdulkabirkhan@gmail.com](mailto:m.abdulkabirkhan@gmail.com).

---

## **📜 License**

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
