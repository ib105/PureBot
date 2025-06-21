# AI Chat App with RAG (Retrieval-Augmented Generation)

A modern AI chat application that supports both general AI conversations and document-based Q&A using RAG technology. Upload PDF documents and ask questions about their content, or chat with the AI assistant directly.

## ✨ Features

- 📄 **PDF Document Upload**: Upload and process PDF documents
- 🧠 **RAG Chat**: Ask questions about uploaded documents
- 🤖 **General AI Chat**: Standard AI assistant conversations
- 💾 **MongoDB Integration**: Store chat history and messages
- 🚀 **Redis Support**: Real-time features (optional)
- 🎨 **Modern UI**: Beautiful, responsive web interface
- 🔍 **Vector Search**: FAISS-based similarity search for documents

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **AI Model**: DeepSeek R1 via OpenRouter. This was used due to cost restrictions in OpenAI
- **Vector Database**: FAISS
- **Embeddings**: HuggingFace all-MiniLM-L6-v2
- **Database**: MongoDB
- **Cache**: Redis (optional)
- **Frontend**: HTML5, CSS3, JavaScript

## 📋 Prerequisites

Before running the application, ensure you have:

- Python 3.8+ installed
- MongoDB running locally or remotely
- Redis running locally (optional)
- OpenRouter API key (for AI responses)

## 🚀 Quick Start

### 1. Enable Virtual Environment
```powershell
venv/Scripts/activate
```

### 2. Clone the Repository

```powershell
git clone <your-repository-url>
cd ai-chat-app
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Environment Setup

Create a `.env` file in the project root:

```env
# OpenRouter API Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here

# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017/
DB_NAME=ai_chat

# Redis Configuration (optional)
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 5. Start Required Services

#### Start MongoDB (Windows):
```powershell
# If MongoDB is installed as a service:
net start MongoDB

# Or start manually:
mongod --dbpath "C:\data\db"
```

#### Start Redis (Windows - Optional):
```powershell
# If Redis is installed as a service:
net start Redis

# Or start manually:
redis-server
```

### 6. Run the Application

```powershell
python app.py
```

The application will start on `http://localhost:5000`

### 7. Access the Application

- **Web Interface**: http://localhost:5000/chat
- **API Documentation**: http://localhost:5000

## 🔧 Configuration

### Required Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key | Required |
| `MONGO_URI` | MongoDB connection string | `mongodb://localhost:27017/` |
| `DB_NAME` | Database name | `ai_chat` |

### Optional Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_HOST` | Redis host | `localhost` |
| `REDIS_PORT` | Redis port | `6379` |

## 📡 API Endpoints

### Upload Document
```powershell
curl -X POST -F "document=@your-file.pdf" http://localhost:5000/api/upload
```

### RAG Chat (Document Q&A)
```powershell
curl -X POST -H "Content-Type: application/json" -d "{\"question\": \"What is this document about?\"}" http://localhost:5000/api/rag
```

### General AI Chat
```powershell
curl -X POST -H "Content-Type: application/json" -d "{\"question\": \"Hello, how are you?\"}" http://localhost:5000/api/ai
```

### Get System Status
```powershell
curl http://localhost:5000/api/status
```

### List Uploaded Documents
```powershell
curl http://localhost:5000/api/documents
```

### Get Message History
```powershell
curl http://localhost:5000/api/messages
```

## 📁 Project Structure

```
ai-chat-app/
├── app.py                 # Main Flask application
├── chat.html             # Frontend interface
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
├── README.md            # This file
└── documents/           # Uploaded PDF files (auto-created)
```

## 🔍 Usage Guide

### 1. Upload a PDF Document
- Click on the upload area in the sidebar
- Select a PDF file (max 10MB)
- Wait for processing to complete

### 2. Ask Questions About Your Document
- Switch to "RAG Chat" mode
- Ask questions about the uploaded document
- The AI will answer based on the document content

### 3. General AI Chat
- Switch to "AI Chat" mode
- Ask any general questions
- The AI will respond without document context

## 🛡️ Error Handling

The application includes comprehensive error handling for:
- File upload failures
- Database connection issues
- API rate limits
- Invalid requests
- Server errors

## 🐛 Troubleshooting

### Common Issues

**1. MongoDB Connection Failed**
```powershell
# Check if MongoDB is running:
net start MongoDB

# Or check the process:
tasklist | findstr mongod
```

**2. OpenRouter API Errors**
- Verify your API key in the `.env` file
- Check your OpenRouter account credits
- Ensure you have access to the DeepSeek model

**3. File Upload Issues**
- Ensure file is a valid PDF
- Check file size (max 10MB)
- Verify upload directory permissions

**4. Redis Connection (Optional)**
```powershell
# Check if Redis is running:
redis-cli ping
```

### Debug Mode

The application runs in debug mode by default. Check the console output for detailed error messages.

## 📊 System Requirements

- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 1GB free space
- **Network**: Internet connection for AI API calls
- **OS**: Windows 10/11, macOS, or Linux

## 🔒 Security Notes

- Keep your OpenRouter API key secure
- Don't commit `.env` files to version control
- Use environment variables for sensitive configuration
- Consider using MongoDB authentication in production

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request


## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section
2. Review the console output for errors
3. Verify all dependencies are installed
4. Ensure all services are running


---

## Chat Interface
<img width="942" alt="image" src="https://github.com/user-attachments/assets/3301ef2f-64ed-4cf1-85ec-756dbdfb6a31" />

**Happy Chatting! 🎉**
