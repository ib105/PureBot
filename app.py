from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv
import redis
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from werkzeug.utils import secure_filename
import os
import json
import datetime
import tempfile
import shutil

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Custom JSON encoder for MongoDB ObjectId
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return super().default(o)

app.json_encoder = JSONEncoder

# File upload configuration
UPLOAD_FOLDER = 'documents'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Get port from environment (Railway sets this)
PORT = int(os.environ.get('PORT', 5000))

# Database connections
try:
    mongo_uri = os.getenv("MONGO_URI") or os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
    client = MongoClient(mongo_uri)
    db = client[os.getenv("DB_NAME", "ai_chat")]
    messages_collection = db["messages"]
    # Test the connection
    client.admin.command('ping')
    print("✅ MongoDB connected")
    mongodb_connected = True
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    messages_collection = None
    mongodb_connected = False

# Redis setup (optional for real-time features)
try:
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    redis_db = int(os.getenv("REDIS_DB", 0))
    
    r = redis.Redis(
        host=redis_host, 
        port=redis_port, 
        db=redis_db, 
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5
    )
    r.ping()  # Test connection
    print("✅ Redis connected")
    redis_connected = True
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
    r = None
    redis_connected = False

# OpenRouter/OpenAI setup
try:
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY")
    
    openai_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    print("✅ OpenRouter client initialized")
    openai_connected = True
except Exception as e:
    print(f"❌ OpenRouter setup failed: {e}")
    openai_client = None
    openai_connected = False

# Global variable to store vectorstore
vectorstore = None
current_document = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_rag_system(pdf_path):
    """Initialize RAG system with uploaded PDF"""
    global vectorstore, current_document
    try:
        print(f"🔄 Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        
        # Split documents into smaller chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        documents = text_splitter.split_documents(pages)
        
        print("🔄 Creating embeddings...")
        # Use HuggingFace embeddings (free, no API key needed)
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # Force CPU usage for Railway
        )
        
        print("🔄 Building vector store...")
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        current_document = {
            'path': pdf_path,
            'filename': os.path.basename(pdf_path),
            'pages': len(pages),
            'chunks': len(documents),
            'uploaded_at': datetime.datetime.now()
        }
        
        print("✅ RAG system initialized successfully")
        return True
    except Exception as e:
        print(f"❌ RAG setup error: {e}")
        vectorstore = None
        current_document = None
        return False

def get_ai_response(question, context=None):
    """Get response from AI model"""
    if not openai_connected or not openai_client:
        return "AI client not configured. Please check your API key.", None
    
    try:
        if context:
            # RAG mode - use context from documents
            system_message = """You are a helpful AI assistant that answers questions based on the provided context from uploaded documents. 
            Use the context to provide accurate and relevant answers. If the context doesn't contain enough information to answer the question, 
            say so clearly. Always be helpful and informative."""
            
            user_message = f"""Context from document:
{context}

Question: {question}

Please answer the question based on the provided context."""
        else:
            # General AI mode
            system_message = "You are a helpful AI assistant. Provide accurate, informative, and helpful responses to user questions."
            user_message = question
        
        response = openai_client.chat.completions.create(
            model="deepseek/deepseek-r1",  # Using DeepSeek R1 via OpenRouter
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=1000,
            temperature=0.7,
        )
        
        answer = response.choices[0].message.content.strip()
        model_used = response.model
        
        return answer, model_used
        
    except Exception as e:
        print(f"❌ AI API error: {e}")
        return f"Sorry, I encountered an error while processing your request: {str(e)}", None

def save_message(message_data):
    """Save message to database"""
    if mongodb_connected and messages_collection is not None:
        try:
            result = messages_collection.insert_one(message_data)
            return str(result.inserted_id)
        except Exception as e:
            print(f"❌ Database save error: {e}")
    return None

def format_message(message):
    """Format message for JSON serialization"""
    if '_id' in message:
        message['_id'] = str(message['_id'])
    if 'timestamp' in message and isinstance(message['timestamp'], datetime.datetime):
        message['timestamp'] = message['timestamp'].isoformat()
    return message

def check_services():
    """Verify all required services are connected"""
    print("\nService Status:")
    
    # Check MongoDB
    if mongodb_connected:
        print(f"   MongoDB: ✅ (Connected to {db.name}.{messages_collection.name})")
    else:
        print("   MongoDB: ❌ (Not connected)")
    
    # Check Redis
    if redis_connected:
        print("   Redis: ✅")
    else:
        print("   Redis: ❌ (Not connected)")
    
    # Check OpenAI/OpenRouter
    if openai_connected:
        print("   OpenRouter: ✅")
    else:
        print("   OpenRouter: ❌ (Not configured)")

# API Endpoints
@app.route('/')
def home():
    return """
    <h1>AI Chat App</h1>
    <p><a href="/chat">Go to Chat Interface</a></p>
    <p>API Endpoints:</p>
    <ul>
        <li>POST /api/messages</li>
        <li>POST /api/ai</li>
        <li>POST /api/rag</li>
        <li>GET /api/stream</li>
    </ul>
    """

@app.route("/chat")
def chat_interface():
    """Serve the chat frontend"""
    try:
        with open('chat.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return """
        <h1>Chat Interface Not Found</h1>
        <p>Please make sure chat.html is in the same directory as app.py</p>
        <p><a href="/">Go back to home</a></p>
        """, 404

@app.route("/api/status", methods=["GET"])
def get_status():
    """Get system status"""
    status = {
        "timestamp": datetime.datetime.now().isoformat(),
        "mongodb_connected": mongodb_connected,
        "redis_connected": redis_connected,
        "ai_client_ready": openai_connected,
        "rag_system_ready": vectorstore is not None,
        "current_document": current_document,
        "status": "healthy"
    }
    return jsonify(status)

@app.route("/api/upload", methods=["POST"])
def upload_document():
    """Handle PDF document upload"""
    if 'document' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['document']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are allowed"}), 400
    
    try:
        # Secure the filename
        filename = secure_filename(file.filename)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        
        # Save the file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Initialize RAG system with the uploaded file
        if initialize_rag_system(filepath):
            return jsonify({
                "status": "Document uploaded and processed successfully!",
                "filename": filename,
                "message": "You can now ask questions about the document",
                "document_info": current_document
            }), 200
        else:
            return jsonify({"error": "Failed to process the uploaded document"}), 500
            
    except Exception as e:
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route("/api/rag", methods=["POST"])
def rag_chat():
    """Handle RAG-based chat queries"""
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({"error": "Question is required"}), 400
    
    question = data['question'].strip()
    
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400
    
    if not vectorstore:
        return jsonify({"error": "No document uploaded. Please upload a PDF document first."}), 400
    
    try:
        # Retrieve relevant documents
        docs = vectorstore.similarity_search(question, k=3)
        
        # Combine context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Get AI response with context
        answer, model_used = get_ai_response(question, context)
        
        # Save conversation to database
        message_data = {
            "question": question,
            "answer": answer,
            "context_used": len(docs),
            "mode": "rag",
            "model": model_used,
            "document": current_document['filename'] if current_document else None,
            "timestamp": datetime.datetime.now()
        }
        
        message_id = save_message(message_data)
        
        response_data = {
            "answer": answer,
            "model": model_used,
            "sources_used": len(docs),
            "document": current_document['filename'] if current_document else None,
            "message_id": message_id
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"error": f"RAG query failed: {str(e)}"}), 500

@app.route("/api/ai", methods=["POST"])
def ai_chat():
    """Handle general AI chat queries"""
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({"error": "Question is required"}), 400
    
    question = data['question'].strip()
    
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400
    
    try:
        # Get AI response without document context
        answer, model_used = get_ai_response(question)
        
        # Save conversation to database
        message_data = {
            "question": question,
            "answer": answer,
            "mode": "ai",
            "model": model_used,
            "timestamp": datetime.datetime.now()
        }
        
        message_id = save_message(message_data)
        
        response_data = {
            "answer": answer,
            "model": model_used,
            "message_id": message_id
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"error": f"AI query failed: {str(e)}"}), 500

@app.route("/api/messages", methods=["GET", "POST"])
def handle_messages():
    """Handle message retrieval and sending"""
    if request.method == "GET":
        # Retrieve recent messages
        try:
            if mongodb_connected and messages_collection is not None:
                messages = list(messages_collection.find().sort("timestamp", -1).limit(50))
                formatted_messages = [format_message(msg) for msg in messages]
                return jsonify({"messages": formatted_messages[::-1]})  # Reverse to show oldest first
            else:
                return jsonify({"messages": []})
        except Exception as e:
            return jsonify({"error": f"Failed to retrieve messages: {str(e)}"}), 500
    
    elif request.method == "POST":
        # This endpoint can be used for simple message sending
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({"error": "Message is required"}), 400
        
        try:
            message_data = {
                "content": data['message'],
                "sender": data.get('sender', 'user'),
                "timestamp": datetime.datetime.now()
            }
            
            message_id = save_message(message_data)
            
            return jsonify({
                "status": "Message saved",
                "message_id": message_id,
                "timestamp": message_data["timestamp"].isoformat()
            })
            
        except Exception as e:
            return jsonify({"error": f"Failed to save message: {str(e)}"}), 500

@app.route("/api/stream")
def stream_messages():
    """Server-sent events endpoint for real-time updates"""
    def event_stream():
        if redis_connected and r is not None:
            pubsub = r.pubsub()
            pubsub.subscribe('chat_channel')
            
            for message in pubsub.listen():
                if message['type'] == 'message':
                    yield f"data: {message['data']}\n\n"
        else:
            yield "data: {\"error\": \"Redis not available\"}\n\n"
    
    return app.response_class(event_stream(), mimetype="text/plain")

@app.route("/api/documents", methods=["GET"])
def list_documents():
    """List uploaded documents"""
    try:
        documents = []
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                if filename.endswith('.pdf'):
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    stat = os.stat(filepath)
                    documents.append({
                        "filename": filename,
                        "size": stat.st_size,
                        "uploaded": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "is_current": current_document and current_document['filename'] == filename
                    })
        
        return jsonify({
            "documents": documents,
            "current_document": current_document
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to list documents: {str(e)}"}), 500

# Health check endpoint for Railway
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "services": {
            "mongodb": mongodb_connected,
            "redis": redis_connected,
            "ai_client": openai_connected
        }
    })

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 10MB."}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    print("🚀 Starting AI Chat App...")
    print("📋 System Status:")
    print(f"   MongoDB: {'✅' if mongodb_connected else '❌'}")
    print(f"   Redis: {'✅' if redis_connected else '❌'}")
    print(f"   OpenRouter: {'✅' if openai_connected else '❌'}")
    print(f"   Upload folder: {UPLOAD_FOLDER}")
    
    # Railway compatibility - use environment variables for host and port
    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    
    print(f"🌐 Server starting on http://{host}:{port}")
    print(f"💬 Chat interface: http://{host}:{port}/chat")
    
    check_services()
    app.run(debug=debug, host=host, port=port)
