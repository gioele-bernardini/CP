import os
import time

def check_modification(file_path):
    # Get the last modification time of the file
    current_modification_time = os.path.getmtime(file_path)
    
    while True:
        # Get the last modification time of the file
        new_modification_time = os.path.getmtime(file_path)
        
        # Check if the modification time has changed
        if new_modification_time != current_modification_time:
            print("The file has been modified!")
            # Update the current modification time
            current_modification_time = new_modification_time
        
        # Wait for a bit before checking again
        time.sleep(1)

# Example usage
file_to_monitor = "buttons.html"
check_modification(file_to_monitor)
