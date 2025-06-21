from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv
import redis
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import datetime
import tempfile
import shutil
import numpy as np
import pickle

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
    # Fixed MongoDB connection string - corrected cluster name
    mongo_uri = os.getenv("MONGO_URI") or os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
    
    # Fix common MongoDB URI issues
    if "Cluster0.mongodb.net" in mongo_uri:
        mongo_uri = mongo_uri.replace("Cluster0.mongodb.net", "cluster0.mongodb.net")
    
    # Add additional connection options for better Railway compatibility
    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=10000,  # Increased timeout
        connectTimeoutMS=20000,
        socketTimeoutMS=30000,
        maxPoolSize=10,
        retryWrites=True,
        tls=True,  # Enable TLS for MongoDB Atlas
        tlsAllowInvalidCertificates=False
    )
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

# Redis setup (optional for real-time features) - Skip in Railway for now
redis_connected = False
r = None
print("ℹ️ Redis skipped for Railway deployment")

# OpenRouter/OpenAI setup - Fixed initialization
try:
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY")
    
    # Fixed OpenAI client initialization - removed unsupported 'proxies' parameter
    openai_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=30.0  # Add timeout for better error handling
    )
    print("✅ OpenRouter client initialized")
    openai_connected = True
except Exception as e:
    print(f"❌ OpenRouter setup failed: {e}")
    openai_client = None
    openai_connected = False

# Global variables for RAG system
document_chunks = []
vectorizer = None
document_vectors = None
current_document = None

# Lightweight Embedding Class using TF-IDF
class LightweightEmbeddings:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000, 
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        self.fitted = False
        self.document_texts = []
    
    def fit_transform(self, texts):
        """Fit the vectorizer and transform texts"""
        self.document_texts = texts
        vectors = self.vectorizer.fit_transform(texts)
        self.fitted = True
        return vectors.toarray()
    
    def transform_query(self, query):
        """Transform a query using the fitted vectorizer"""
        if not self.fitted:
            return np.zeros(1000)
        return self.vectorizer.transform([query]).toarray()[0]
    
    def similarity_search(self, query, k=3):
        """Find most similar documents to query"""
        if not self.fitted or not self.document_texts:
            return []
        
        query_vector = self.transform_query(query)
        
        # Calculate cosine similarity
        similarities = cosine_similarity([query_vector], document_vectors)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Return documents with similarity scores
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                results.append({
                    'content': self.document_texts[idx],
                    'similarity': similarities[idx],
                    'index': idx
                })
        
        return results

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_rag_system(pdf_path):
    """Initialize RAG system with uploaded PDF"""
    global document_chunks, vectorizer, document_vectors, current_document
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
        # Extract text content from documents
        texts = [doc.page_content for doc in documents]
        
        # Initialize lightweight embeddings
        vectorizer = LightweightEmbeddings()
        document_vectors = vectorizer.fit_transform(texts)
        document_chunks = documents
        
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
        document_chunks = []
        vectorizer = None
        document_vectors = None
        current_document = None
        return False

def get_ai_response(question, context=None):
    """Get response from AI model with improved error handling"""
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
        
        # Add retry logic for API calls
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = openai_client.chat.completions.create(
                    model=os.getenv("AI_MODEL", "deepseek/deepseek-r1"),
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=int(os.getenv("MAX_TOKENS", 1000)),
                    temperature=float(os.getenv("TEMPERATURE", 0.7)),
                )
                
                answer = response.choices[0].message.content.strip()
                model_used = response.model
                
                return answer, model_used
                
            except Exception as retry_error:
                if attempt == max_retries - 1:
                    raise retry_error
                print(f"API attempt {attempt + 1} failed, retrying...")
                continue
        
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

# API Endpoints
@app.route('/')
def home():
    return """
    <h1>🤖 AI Chat App</h1>
    <p><strong><a href="/chat">🚀 Go to Chat Interface</a></strong></p>
    <h2>📡 API Endpoints:</h2>
    <ul>
        <li><strong>POST /api/messages</strong> - Send/retrieve messages</li>
        <li><strong>POST /api/ai</strong> - General AI chat</li>
        <li><strong>POST /api/rag</strong> - RAG-based document chat</li>
        <li><strong>POST /api/upload</strong> - Upload PDF documents</li>
        <li><strong>GET /api/status</strong> - System status</li>
        <li><strong>GET /health</strong> - Health check</li>
    </ul>
    <h2>🔧 System Status:</h2>
    <ul>
        <li>MongoDB: {"✅ Connected" if mongodb_connected else "❌ Disconnected"}</li>
        <li>Redis: {"✅ Connected" if redis_connected else "❌ Disconnected"}</li>
        <li>AI Client: {"✅ Ready" if openai_connected else "❌ Not configured"}</li>
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
        "rag_system_ready": vectorizer is not None and len(document_chunks) > 0,
        "current_document": current_document,
        "status": "healthy" if (mongodb_connected or openai_connected) else "degraded"
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
    
    if not vectorizer or not document_chunks:
        return jsonify({"error": "No document uploaded. Please upload a PDF document first."}), 400
    
    try:
        # Retrieve relevant documents using lightweight similarity search
        similar_docs = vectorizer.similarity_search(question, k=3)
        
        if not similar_docs:
            return jsonify({"error": "No relevant content found in the document."}), 400
        
        # Combine context from retrieved documents
        context = "\n\n".join([doc['content'] for doc in similar_docs])
        
        # Get AI response with context
        answer, model_used = get_ai_response(question, context)
        
        # Save conversation to database
        message_data = {
            "question": question,
            "answer": answer,
            "context_used": len(similar_docs),
            "mode": "rag",
            "model": model_used,
            "document": current_document['filename'] if current_document else None,
            "timestamp": datetime.datetime.now()
        }
        
        message_id = save_message(message_data)
        
        response_data = {
            "answer": answer,
            "model": model_used,
            "sources_used": len(similar_docs),
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
    
    # Show startup warnings
    if not mongodb_connected:
        print("⚠️  Warning: MongoDB not connected - message persistence disabled")
    if not redis_connected:
        print("⚠️  Warning: Redis not connected - real-time features disabled")
    if not openai_connected:
        print("⚠️  Warning: AI client not ready - chat features disabled")
    
    app.run(debug=debug, host=host, port=port)
