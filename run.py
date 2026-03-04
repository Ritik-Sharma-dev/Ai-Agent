import subprocess
import sys
import time

def start_servers():
    print("🚀 Booting up the TalentScout AI Suite...")
    
    # 1. Start the FastAPI Backend
    print("⚙️ Starting FastAPI Backend on port 8000...")
    backend = subprocess.Popen([sys.executable, "backend/main.py"])
    
    # Give the backend 2 seconds to fully start up so the frontend doesn't crash trying to connect
    time.sleep(2)
    
    # 2. Start the Streamlit Frontend
    print("🎨 Starting Streamlit Frontend...")
    # FIX: Point exactly to the main search page
    frontend = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "frontend/2_Candidate_Coach.py"])
    
    try:
        # Keep the script running while both servers are active
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        # If you press Ctrl+C, it cleanly shuts down BOTH servers at once
        print("\n🛑 Shutting down all servers...")
        backend.terminate()
        frontend.terminate()
        print("✅ Servers successfully stopped.")

if __name__ == "__main__":
    start_servers()