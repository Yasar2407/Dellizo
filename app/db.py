from pymongo import MongoClient, errors
from .config import MONGODB_URI
import certifi

# Connect to MongoDB Atlas with proper TLS/SSL certs
try:
    client = MongoClient(MONGODB_URI, tls=True, tlsCAFile=certifi.where()) 
    db = client["emotiondb"]
    analyses_coll = db["analyses"]

    # Test connection
    client.admin.command("ping")
    print("✅ MongoDB connection successful!")

except errors.ServerSelectionTimeoutError as err:
    print("❌ MongoDB connection failed:", err)
    raise err

# Create basic indexes safely
def ensure_indexes():
    try:
        analyses_coll.create_index("predicted_emotion")
        analyses_coll.create_index("timestamp")
        print("✅ Indexes ensured!")
    except Exception as e:
        print("❌ Error creating indexes:", e)
        raise e
