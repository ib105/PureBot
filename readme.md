# PureBot

A powerful AI-powered chat application with RAG (Retrieval-Augmented Generation) capabilities that allows users to upload PDF documents and ask questions about their content using advanced AI models.

**Please Note that this bot was deployed in Render's free tier for testing so loading will take around 50 seconds.**



## Features

- **PDF Document Upload**: Upload and process PDF documents up to 10MB
- **RAG-Powered Chat**: Ask questions about uploaded documents with context-aware responses
- **General AI Chat**: Chat with AI assistant for general queries
- **Smart Document Search**: TF-IDF based similarity search for relevant content retrieval
- **Responsive Design**: Beautiful, mobile-friendly interface
- **Real-time Responses**: Fast AI responses powered by OpenRouter/DeepSeek
- **Message Persistence**: MongoDB integration for chat history
- **Modern UI**: Gradient backgrounds, animations, and intuitive design

## Technology Stack

### Backend
- **Flask** - Python web framework
- **OpenRouter/OpenAI** - AI model integration
- **MongoDB** - Database for message persistence
- **LangChain** - Document processing and text splitting
- **scikit-learn** - TF-IDF vectorization and similarity search
- **PyPDF** - PDF document parsing

### Frontend
- **Vanilla JavaScript** - Interactive chat interface
- **CSS3** - Modern styling with gradients and animations
- **HTML5** - Semantic markup

### Deployment
- **Render** - Cloud deployment platform
- **Gunicorn** - WSGI HTTP Server

## Quick Start

### Prerequisites

- Python 3.8+
- MongoDB Atlas account (or local MongoDB)
- OpenRouter API key (or OpenAI API key)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-chat-app.git
   cd ai-chat-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   MONGO_URI=your_mongodb_atlas_connection_string
   DB_NAME=ai_chat
   AI_MODEL=deepseek/deepseek-r1
   MAX_TOKENS=1000
   TEMPERATURE=0.7
   ```

4. **Create upload directory**
   ```bash
   mkdir documents
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   - Main interface: `http://localhost:5000`
   - Chat interface: `http://localhost:5000/chat`

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENROUTER_API_KEY` | OpenRouter API key for AI models | - | Yes |
| `OPENAI_API_KEY` | Alternative to OpenRouter | - | No |
| `MONGO_URI` | MongoDB connection string | `mongodb://localhost:27017/` | No |
| `DB_NAME` | Database name | `ai_chat` | No |
| `AI_MODEL` | AI model to use | `deepseek/deepseek-r1` | No |
| `MAX_TOKENS` | Maximum response tokens | `1000` | No |
| `TEMPERATURE` | AI response creativity | `0.7` | No |
| `PORT` | Server port | `5000` | No |
| `HOST` | Server host | `0.0.0.0` | No |

### Supported AI Models

The app supports various models through OpenRouter:
- `deepseek/deepseek-r1` (default)
- `openai/gpt-3.5-turbo`
- `openai/gpt-4`
- `anthropic/claude-3-haiku`
- And many more available on OpenRouter

## 📡 API Endpoints

### Chat Endpoints
- `POST /api/ai` - General AI chat
- `POST /api/rag` - RAG-based document chat
- `POST /api/upload` - Upload PDF documents

### Utility Endpoints
- `GET /api/status` - System status check
- `GET /api/messages` - Retrieve chat history
- `POST /api/messages` - Save messages
- `GET /health` - Health check

### API Examples

**General AI Chat:**
```bash
curl -X POST http://localhost:5000/api/ai \
  -H "Content-Type: application/json" \
  -d '{"question": "What is artificial intelligence?"}'
```

**RAG Document Chat:**
```bash
curl -X POST http://localhost:5000/api/rag \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the uploaded document?"}'
```

**Document Upload:**
```bash
curl -X POST http://localhost:5000/api/upload \
  -F "document=@your_document.pdf"
```

## Architecture

### RAG System Flow

1. **Document Upload** → PDF parsing with PyPDF
2. **Text Splitting** → Chunking with LangChain
3. **Vectorization** → TF-IDF embeddings with scikit-learn
4. **Query Processing** → Similarity search for relevant chunks
5. **Context Injection** → AI response with document context

### File Structure

```
ai-chat-app/
├── app.py              # Main Flask application
├── chat.html           # Frontend interface
├── requirements.txt    # Python dependencies
├── render.yaml         # Render deployment config
├── .env               # Environment variables (create this)
├── documents/         # PDF uploads directory
└── README.md          # This file
```

## Deployment

### Deploy on Render

1. **Fork this repository**

2. **Create a new Web Service on Render**
   - Connect your GitHub repository
   - Use the provided `render.yaml` configuration

3. **Set environment variables in Render dashboard:**
   - `OPENROUTER_API_KEY`
   - `MONGO_URI`
   - `DB_NAME`

4. **Deploy automatically**
   - Render will build and deploy your app
   - Access via the provided Render URL


## Usage

### Basic Chat Flow

1. **Visit the chat interface** at `/chat`
2. **Upload a PDF document** using the drag-and-drop area
3. **Wait for processing** - you'll see a success message
4. **Ask questions** about your document in RAG mode
5. **Switch to AI mode** for general questions

### Example Queries

**Document-specific (RAG mode):**
- "What is the main argument in this paper?"
- "Summarize the key findings"
- "What methodology was used?"

**General AI (AI mode):**
- "Explain quantum computing"
- "Write a Python function to sort a list"
- "What's the weather like today?"

## Security Features

- **File Type Validation** - Only PDF files allowed
- **File Size Limits** - Maximum 10MB uploads
- **Secure Filename Handling** - Protection against path traversal
- **Environment Variable Protection** - Sensitive data in .env
- **Error Handling** - Graceful failure modes

## Troubleshooting

### Common Issues

**MongoDB Connection Failed:**
```bash
# Check your connection string format
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/database?retryWrites=true&w=majority
```

**AI API Errors:**
```bash
# Verify your API key
curl -H "Authorization: Bearer YOUR_API_KEY" https://openrouter.ai/api/v1/models
```

**File Upload Issues:**
- Ensure `documents/` directory exists
- Check file size (max 10MB)
- Verify PDF format

**Performance Issues:**
- MongoDB Atlas M0 (free tier) has connection limits
- Consider upgrading for production use
- Monitor API usage and rate limits

### Debug Mode

Enable debug mode for development:
```bash
export FLASK_DEBUG=true
python app.py
```

### App Interface
<img width="895" alt="image" src="https://github.com/user-attachments/assets/cabf1e21-3f7f-4c4a-8e66-4b6b337c1a4b" />


## Contributing

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 for Python code
- Add docstrings to functions
- Test with different PDF types
- Ensure mobile responsiveness
- Add error handling for new features

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenRouter](https://openrouter.ai/) for AI model access
- [LangChain](https://langchain.com/) for document processing
- [MongoDB Atlas](https://www.mongodb.com/atlas) for database hosting
- [Render](https://render.com/) for deployment platform


