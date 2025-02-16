import streamlit.web.cli as stcli
import sys
import os

if __name__ == "__main__":
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up the command line arguments for Streamlit
    sys.argv = [
        "streamlit",
        "run",
        os.path.join(current_dir, "app.py"),
        "--server.address=0.0.0.0",  # Allow external connections
        "--server.port=8501",        # Default Streamlit port
        "--browser.serverAddress=localhost",  # Use localhost for browser
    ]
    
    # Run Streamlit
    sys.exit(stcli.main()) 