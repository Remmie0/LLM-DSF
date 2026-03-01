import os
import shutil
from datetime import datetime

### Configuration
archive_name = "knmi_ablation"

## Functions
def ensure_archive_folder(root_folder):
    """Create the archive folder if it doesn't exist."""
    archive_folder = os.path.join(root_folder, 'archive')
    if not os.path.exists(archive_folder):
        os.makedirs(archive_folder)
    return archive_folder

def create_archive_subfolder(archive_folder, extra_name=None):
    """Create a timestamped archive subfolder."""
    timestamp = datetime.now().strftime('%Y-%m-%d')
    folder_name = f"archive-{timestamp}"

    if extra_name:
        folder_name += f"-{extra_name}"

    archive_path = os.path.join(archive_folder, folder_name)

    if not os.path.exists(archive_path):
        os.makedirs(archive_path)

    return archive_path

def is_log_file(filename):
    """Check if a file is a log file."""
    return filename.endswith('.log')

def archive_logs(root_folder, extra_name=None):
    """Archive all log files from the root folder."""
    try:
        # Ensure archive folder exists
        archive_folder = ensure_archive_folder(root_folder)

        # Create new archive subfolder
        archive_path = create_archive_subfolder(archive_folder, extra_name)

        # Get all log files in root folder
        log_files = [f for f in os.listdir(root_folder)
                    if os.path.isfile(os.path.join(root_folder, f))
                    and is_log_file(f)]

        if not log_files:
            print("No log files found to archive")
            return

        # Copy and empty each log file
        for log_file in log_files:
            source_path = os.path.join(root_folder, log_file)
            dest_path = os.path.join(archive_path, log_file)

            # Copy the log file
            shutil.copy2(source_path, dest_path)
            print(f"Archived {log_file}")

            # Empty the original log file
            with open(source_path, 'w') as f:
                f.truncate(0)
            print(f"Emptied {log_file}")

        print(f"Successfully archived {len(log_files)} log files")

    except Exception as e:
        print(f"Error during archiving: {str(e)}")
        raise

def main():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Archive the logs
    # You can pass an extra_name parameter if needed
    # archive_logs(script_dir, extra_name="backup")
    archive_logs(script_dir, extra_name=archive_name)

if __name__ == "__main__":
    main()
