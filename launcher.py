import webbrowser
import subprocess
import time
import threading
import os

def start_streamlit():
    subprocess.Popen(["streamlit", "run", "main.py"])

def open_landing_page():
    path = os.path.abspath("frontend/index.html")
    webbrowser.open(f"file://{path}")

if __name__ == "__main__":
    threading.Thread(target=start_streamlit).start()
    time.sleep(2)
    open_landing_page()
