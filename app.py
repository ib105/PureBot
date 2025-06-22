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
import io
from gridfs import GridFS

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
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Get port from environment (Render sets this)
PORT = int(os.environ.get('PORT', 5000))

# Database connections
try:
    # MongoDB connection string
    mongo_uri = os.getenv("MONGO_URI") or os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
    
    # Fix common MongoDB URI issues
    if "Cluster0.mongodb.net" in mongo_uri:
        mongo_uri = mongo_uri.replace("Cluster0.mongodb.net", "cluster0.mongodb.net")
    
    # Add additional connection options for better cloud compatibility
    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=10000,
        connectTimeoutMS=20000,
        socketTimeoutMS=30000,
        maxPoolSize=10,
        retryWrites=True,
        tls=True,
        tlsAllowInvalidCertificates=False
    )
    db = client[os.getenv("DB_NAME", "ai_chat")]
    messages_collection = db["messages"]
    documents_collection = db["documents"]  # Store document metadata
    embeddings_collection = db["embeddings"]  # Store vector embeddings
    
    # Initialize GridFS for file storage
    fs = GridFS(db)
    
    # Test the connection
    client.admin.command('ping')
    print("✅ MongoDB connected")
    mongodb_connected = True
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    messages_collection = None
    documents_collection = None
    embeddings_collection = None
    fs = None
    mongodb_connected = False

# Redis setup (optional) - Skip for cloud deployment
redis_connected = False
r = None
print("ℹ️ Redis skipped for cloud deployment")

# OpenRouter/OpenAI setup
try:
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY")
    
    openai_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=30.0
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
        self.document_vectors = None  # Store vectors in the class instance
    
    def fit_transform(self, texts):
        """Fit the vectorizer and transform texts"""
        self.document_texts = texts
        vectors = self.vectorizer.fit_transform(texts)
        self.fitted = True
        self.document_vectors = vectors.toarray()  # Store in instance
        return self.document_vectors
    
    def transform_query(self, query):
        """Transform a query using the fitted vectorizer"""
        if not self.fitted:
            return np.zeros(1000)
        return self.vectorizer.transform([query]).toarray()[0]
    
    def similarity_search(self, query, k=3):
        """Find most similar documents to query"""
        if not self.fitted or not self.document_texts or self.document_vectors is None:
            print("❌ Embeddings not fitted or no document texts")
            print(f"   fitted: {self.fitted}")
            print(f"   document_texts: {len(self.document_texts) if self.document_texts else 0}")
            print(f"   document_vectors: {self.document_vectors is not None}")
            return []
        
        try:
            query_vector = self.transform_query(query)
            print(f"🔍 Query: '{query}'")
            print(f"🔍 Query vector shape: {query_vector.shape}")
            print(f"📄 Document vectors shape: {self.document_vectors.shape}")
            
            # Calculate cosine similarity using instance vectors
            similarities = cosine_similarity([query_vector], self.document_vectors)[0]
            print(f"📊 Similarity scores: min={similarities.min():.3f}, max={similarities.max():.3f}, mean={similarities.mean():.3f}")
            
            # Get top k indices
            top_indices = np.argsort(similarities)[::-1][:k]
            
            # Return documents with similarity scores (very low threshold)
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.001:  # Very low threshold
                    results.append({
                        'content': self.document_texts[idx],
                        'similarity': similarities[idx],
                        'index': idx
                    })
                    print(f"✅ Found relevant chunk {idx} with similarity {similarities[idx]:.3f}")
            
            # If no results above threshold, return top results anyway
            if not results:
                print("⚠️ No results above threshold, returning top results")
                for i, idx in enumerate(top_indices[:3]):
                    results.append({
                        'content': self.document_texts[idx],
                        'similarity': similarities[idx],
                        'index': idx
                    })
                    print(f"📝 Fallback chunk {idx} with similarity {similarities[idx]:.3f}")
            
            return results
        
        except Exception as e:
            print(f"❌ Similarity search error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def save_to_db(self, document_id):
        """Save vectorizer and embeddings to MongoDB"""
        if not mongodb_connected or not self.fitted:
            return False
        
        try:
            # Save vectorizer
            vectorizer_data = {
                'document_id': document_id,
                'vectorizer': pickle.dumps(self.vectorizer),
                'document_texts': self.document_texts,
                'vectors': self.document_vectors.tolist(),  # Use instance vectors
                'created_at': datetime.datetime.now()
            }
            
            # Remove existing embeddings for this document
            embeddings_collection.delete_many({'document_id': document_id})
            
            # Insert new embeddings
            embeddings_collection.insert_one(vectorizer_data)
            print(f"✅ Embeddings saved for document {document_id}")
            print(f"   Texts: {len(self.document_texts)}")
            print(f"   Vectors: {self.document_vectors.shape}")
            return True
        except Exception as e:
            print(f"❌ Failed to save embeddings: {e}")
            return False
    
    def load_from_db(self, document_id):
        """Load vectorizer and embeddings from MongoDB"""
        if not mongodb_connected:
            print("❌ MongoDB not connected")
            return False
        
        try:
            embedding_data = embeddings_collection.find_one({'document_id': document_id})
            if not embedding_data:
                print(f"❌ No embeddings found for document {document_id}")
                return False
            
            self.vectorizer = pickle.loads(embedding_data['vectorizer'])
            self.document_texts = embedding_data['document_texts']
            self.fitted = True
            
            # Store vectors in instance, not global variable
            self.document_vectors = np.array(embedding_data['vectors'])
            
            print(f"✅ Embeddings loaded for document {document_id}")
            print(f"📊 Loaded {len(self.document_texts)} text chunks")
            print(f"📊 Vector shape: {self.document_vectors.shape}")
            print(f"📊 Vectorizer vocabulary size: {len(self.vectorizer.vocabulary_)}")
            return True
        except Exception as e:
            print(f"❌ Failed to load embeddings: {e}")
            import traceback
            traceback.print_exc()
            return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_file_to_gridfs(file_data, filename, content_type):
    """Save file to MongoDB GridFS"""
    if not mongodb_connected or not fs:
        return None
    
    try:
        file_id = fs.put(
            file_data,
            filename=filename,
            content_type=content_type,
            upload_date=datetime.datetime.now()
        )
        return file_id
    except Exception as e:
        print(f"❌ GridFS save error: {e}")
        return None

def get_file_from_gridfs(file_id):
    """Retrieve file from MongoDB GridFS"""
    if not mongodb_connected or not fs:
        return None
    
    try:
        return fs.get(file_id)
    except Exception as e:
        print(f"❌ GridFS retrieve error: {e}")
        return None

def initialize_rag_system(file_data, filename):
    """Initialize RAG system with uploaded PDF data"""
    global document_chunks, vectorizer, current_document
    
    try:
        print(f"🔄 Processing PDF: {filename}")
        
        # Create a temporary file for PyPDFLoader
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(file_data)
            temp_path = temp_file.name
        
        try:
            # Load and process PDF
            loader = PyPDFLoader(temp_path)
            pages = loader.load_and_split()
            
            if not pages:
                print("❌ No pages extracted from PDF")
                return False
            
            print(f"📄 Extracted {len(pages)} pages")
            
            # Split documents into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            
            documents = text_splitter.split_documents(pages)
            
            if not documents:
                print("❌ No chunks created from documents")
                return False
            
            print(f"📝 Created {len(documents)} chunks")
            
            # Extract text content from documents
            texts = [doc.page_content for doc in documents]
            
            # Filter out empty texts
            texts = [text.strip() for text in texts if text.strip()]
            
            if not texts:
                print("❌ No valid text content found")
                return False
            
            print(f"✅ Processing {len(texts)} text chunks")
            
            # Initialize lightweight embeddings
            vectorizer = LightweightEmbeddings()
            vectorizer.fit_transform(texts)
            document_chunks = documents
            
            print("✅ Embeddings created successfully")
            
            # Save file to GridFS
            file_id = save_file_to_gridfs(file_data, filename, 'application/pdf')
            
            if file_id and mongodb_connected:
                # Save document metadata
                doc_metadata = {
                    'filename': filename,
                    'file_id': file_id,
                    'pages': len(pages),
                    'chunks': len(documents),
                    'uploaded_at': datetime.datetime.now(),
                    'file_size': len(file_data)
                }
                
                # Remove existing document with same filename
                documents_collection.delete_many({'filename': filename})
                
                # Insert new document
                doc_result = documents_collection.insert_one(doc_metadata)
                document_id = doc_result.inserted_id
                
                # Save embeddings
                vectorizer.save_to_db(document_id)
                
                current_document = {
                    'id': str(document_id),
                    'filename': filename,
                    'pages': len(pages),
                    'chunks': len(documents),
                    'uploaded_at': datetime.datetime.now()
                }
            
            print("✅ RAG system initialized successfully")
            return True
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
                
    except Exception as e:
        print(f"❌ RAG setup error: {e}")
        import traceback
        traceback.print_exc()
        document_chunks = []
        vectorizer = None
        current_document = None
        return False

def load_existing_document():
    """Load the most recent document from database"""
    global document_chunks, vectorizer, current_document
    
    if not mongodb_connected:
        return False
    
    try:
        # Get the most recent document
        doc_metadata = documents_collection.find_one(sort=[('uploaded_at', -1)])
        if not doc_metadata:
            print("📭 No existing documents found")
            return False
        
        document_id = doc_metadata['_id']
        print(f"🔄 Loading document: {doc_metadata['filename']}")
        
        # Load embeddings
        vectorizer = LightweightEmbeddings()
        if vectorizer.load_from_db(document_id):
            # Reconstruct document chunks (simplified)
            document_chunks = [type('obj', (object,), {'page_content': text}) for text in vectorizer.document_texts]
            
            current_document = {
                'id': str(document_id),
                'filename': doc_metadata['filename'],
                'pages': doc_metadata['pages'],
                'chunks': doc_metadata['chunks'],
                'uploaded_at': doc_metadata['uploaded_at']
            }
            
            print(f"✅ Loaded existing document: {doc_metadata['filename']}")
            return True
        else:
            print(f"❌ Failed to load embeddings for document: {doc_metadata['filename']}")
    
    except Exception as e:
        print(f"❌ Failed to load existing document: {e}")
        import traceback
        traceback.print_exc()
    
    return False

# Load existing document on startup
load_existing_document()

def get_ai_response(question, context=None):
    """Get response from AI model with improved error handling"""
    if not openai_connected or not openai_client:
        return "AI client not configured. Please check your API key.", None
    
    try:
        if context:
            system_message = """You are a helpful AI assistant that answers questions based on the provided context from uploaded documents. 
            Use the context to provide accurate and relevant answers. If the context doesn't contain enough information to answer the question, 
            say so clearly. Always be helpful and informative."""
            
            user_message = f"""Context from document:
{context}

Question: {question}

Please answer the question based on the provided context."""
        else:
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
    return f"""
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
        <li>Current Document: {"✅ " + current_document['filename'] if current_document else "❌ None"}</li>
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
        
        # Read file data into memory
        file_data = file.read()
        
        # Initialize RAG system with the file data
        if initialize_rag_system(file_data, filename):
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
    
    # Check RAG system status
    if not vectorizer:
        return jsonify({"error": "No document processing system available. Please upload a PDF document first."}), 400
    
    if not vectorizer.fitted:
        return jsonify({"error": "Document processing system not ready. Please try uploading the document again."}), 400
    
    if not document_chunks:
        return jsonify({"error": "No document chunks available. Please upload a PDF document first."}), 400
    
    try:
        print(f"🔍 Processing RAG query: '{question}'")
        print(f"📚 Available chunks: {len(document_chunks)}")
        print(f"🔧 Vectorizer fitted: {vectorizer.fitted}")
        print(f"📄 Document texts count: {len(vectorizer.document_texts) if vectorizer.document_texts else 0}")
        print(f"🧮 Document vectors shape: {vectorizer.document_vectors.shape if vectorizer.document_vectors is not None else 'None'}")
        
        # Retrieve relevant documents using similarity search
        similar_docs = vectorizer.similarity_search(question, k=5)
        
        print(f"🔍 Found {len(similar_docs)} similar documents")
        
        if not similar_docs:
            # Fallback: use first few chunks
            print("⚠️ No similar docs found, using first 3 chunks as fallback")
            if len(document_chunks) >= 3:
                fallback_chunks = document_chunks[:3]
            else:
                fallback_chunks = document_chunks
            
            context = "\n\n".join([chunk.page_content for chunk in fallback_chunks])
            sources_used = len(fallback_chunks)
            
            if not context.strip():
                return jsonify({"error": "Document appears to be empty or unreadable. Please try uploading again."}), 400
        else:
            # Combine context from retrieved documents
            context = "\n\n".join([doc['content'] for doc in similar_docs])
            sources_used = len(similar_docs)
            print(f"✅ Using {sources_used} relevant chunks")
        
        print(f"📝 Context length: {len(context)} characters")
        
        # Get AI response with context
        answer, model_used = get_ai_response(question, context)
        
        if not answer:
            return jsonify({"error": "Failed to generate AI response. Please try again."}), 500
        
        # Save conversation to database
        message_data = {
            "question": question,
            "answer": answer,
            "context_used": sources_used,
            "mode": "rag",
            "model": model_used,
            "document": current_document['filename'] if current_document else None,
            "timestamp": datetime.datetime.now()
        }
        
        message_id = save_message(message_data)
        
        response_data = {
            "answer": answer,
            "model": model_used,
            "sources_used": sources_used,
            "document": current_document['filename'] if current_document else None,
            "message_id": message_id
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ RAG query failed: {e}")
        import traceback
        traceback.print_exc()
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

@app.route("/api/debug/rag", methods=["GET"])
def debug_rag_detailed():
    """Comprehensive RAG system debug info"""
    debug_info = {
        "timestamp": datetime.datetime.now().isoformat(),
        "vectorizer_exists": vectorizer is not None,
        "document_chunks_count": len(document_chunks),
        "current_document": current_document
    }
    
    if vectorizer:
        debug_info.update({
            "vectorizer_fitted": vectorizer.fitted,
            "document_texts_count": len(vectorizer.document_texts) if vectorizer.document_texts else 0,
            "document_vectors_shape": vectorizer.document_vectors.shape if vectorizer.document_vectors is not None else None,
            "has_sklearn_vectorizer": hasattr(vectorizer, 'vectorizer') and vectorizer.vectorizer is not None
        })
        
        if vectorizer.fitted and hasattr(vectorizer.vectorizer, 'vocabulary_'):
            debug_info["vectorizer_vocab_size"] = len(vectorizer.vectorizer.vocabulary_)
            # Sample some vocabulary words
            vocab_sample = list(vectorizer.vectorizer.vocabulary_.keys())[:10]
            debug_info["vocab_sample"] = vocab_sample
    
    # Check database contents
    if mongodb_connected:
        try:
            docs_count = documents_collection.count_documents({})
            embeddings_count = embeddings_collection.count_documents({})
            
            debug_info.update({
                "mongodb_connected": True,
                "db_documents_count": docs_count,
                "db_embeddings_count": embeddings_count
            })
            
            # Get latest document from DB
            latest_doc = documents_collection.find_one(sort=[('uploaded_at', -1)])
            if latest_doc:
                debug_info["latest_db_document"] = {
                    "id": str(latest_doc['_id']),
                    "filename": latest_doc['filename'],
                    "chunks": latest_doc['chunks'],
                    "uploaded_at": latest_doc['uploaded_at'].isoformat()
                }
        except Exception as e:
            debug_info["db_error"] = str(e)
    else:
        debug_info["mongodb_connected"] = False
    
    return jsonify(debug_info)

@app.route("/api/debug/search", methods=["POST"])
def debug_search():
    """Test search functionality with detailed output"""
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Query required"}), 400
    
    query = data['query']
    
    if not vectorizer:
        return jsonify({"error": "No vectorizer available"}), 400
    
    if not vectorizer.fitted:
        return jsonify({"error": "Vectorizer not fitted"}), 400
    
    try:
        # Detailed search test
        results = vectorizer.similarity_search(query, k=5)
        
        response = {
            "query": query,
            "results_count": len(results),
            "vectorizer_status": {
                "fitted": vectorizer.fitted,
                "document_texts_count": len(vectorizer.document_texts),
                "vectors_shape": vectorizer.document_vectors.shape if vectorizer.document_vectors is not None else None
            }
        }
        
        if results:
            response["results"] = [
                {
                    "index": r['index'],
                    "similarity": float(r['similarity']),
                    "content_length": len(r['content']),
                    "content_preview": r['content'][:200] + "..." if len(r['content']) > 200 else r['content']
                } for r in results
            ]
        else:
            response["fallback_info"] = {
                "total_texts": len(vectorizer.document_texts),
                "first_text_preview": vectorizer.document_texts[0][:200] + "..." if vectorizer.document_texts else "No texts available"
            }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"Search test failed: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.route("/api/messages", methods=["GET", "POST"])
def handle_messages():
    """Handle message retrieval and sending"""
    if request.method == "GET":
        try:
            if mongodb_connected and messages_collection is not None:
                messages = list(messages_collection.find().sort("timestamp", -1).limit(50))
                formatted_messages = [format_message(msg) for msg in messages]
                return jsonify({"messages": formatted_messages[::-1]})
            else:
                return jsonify({"messages": []})
        except Exception as e:
            return jsonify({"error": f"Failed to retrieve messages: {str(e)}"}), 500
    
    elif request.method == "POST":
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

# Health check endpoint
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
    
    if current_document:
        print(f"   Current Document: ✅ {current_document['filename']}")
    else:
        print("   Current Document: ❌ None")
    
    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    
    print(f"🌐 Server starting on http://{host}:{port}")
    print(f"💬 Chat interface: http://{host}:{port}/chat")
    
    if not mongodb_connected:
        print("⚠️  Warning: MongoDB not connected - features limited")
    if not openai_connected:
        print("⚠️  Warning: AI client not ready - chat features disabled")
    
    app.run(debug=debug, host=host, port=port)
