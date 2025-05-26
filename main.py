import tkinter as tk
import os
import sys

# Add the parent directory of 'main.py' to the Python path
# This allows importing modules from 'utils' and 'models' directories
script_dir = os.path.dirname(__file__)
sys.path.append(script_dir)

# Now you can import from your subdirectories
# from utils.dataloader import TrafficDataLoader
# from models.saemodel import StackedAutoencoder
from gui.app import TFPSApp # Import the GUI application

def main():
    """
    The main function to run the Traffic Flow Prediction System.
    It initializes and starts the GUI application.
    """
    print("Starting Traffic Flow Prediction System...")

    # Create the main Tkinter window
    root = tk.Tk()

    # Initialize and run the GUI application
    app = TFPSApp(root)

    # Start the Tkinter event loop
    root.mainloop()

    print("Traffic Flow Prediction System exited.")

if __name__ == "__main__":
    # Ensure the directory structure is set up for imports
    # Create dummy directories if they don't exist for local testing
    if not os.path.exists('utils'):
        os.makedirs('utils')
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('gui'):
        os.makedirs('gui')

    # You might want to place your 'Scats Data October 2006.csv'
    # in the root directory where main.py is, or adjust the path in dataloader.py

    main()