import sys
from pathlib import Path
import webbrowser
import time
import os

def check_environment():
    try:
        import dash
        import pandas as pd
        import plotly
        return True
    except ImportError as e:
        print(f"Error: Missing required package - {str(e)}")
        print("Please run: pip install -r requirements-core.txt")
        return False

def check_data_file():
    data_file = Path('Astros.data/Houston Astros Roster Data.csv')
    if not data_file.exists():
        print(f"Error: Data file not found at {data_file}")
        return False
    return True

def main():
    if not check_environment():
        sys.exit(1)
    
    if not check_data_file():
        sys.exit(1)
    
    print("Starting Dash server...")
    print("The dashboard will open in your default web browser...")
    
    # Import dashboard after checks to avoid potential import errors
    from dashboard import app
    
    # Get port from environment or use 8052 as fallback
    port = int(os.environ.get('PORT', 8052))
    print(f"Dash is running on http://127.0.0.1:{port}/")

    webbrowser.open(f'http://127.0.0.1:{port}/')
    
    # Run the server
    app.run_server(debug=True, port=port)

if __name__ == '__main__':
    main() 