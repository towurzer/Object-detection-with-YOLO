from roboflow import Roboflow
import os

# Your API key
api_key = os.environ.get('ROBOFLOW_API_KEY')
if not api_key:
    print("Please set ROBOFLOW_API_KEY environment variable")
    exit(1)

rf = Roboflow(Can9PbK69yyBXwMWG1OW)

# You need to specify your workspace and project
# Common workspaces: your username or organization name
print("Available workspaces:", rf.workspaces())

# Replace these with your actual values
WORKSPACE = "your-workspace-name"  # <-- Needs to be set
PROJECT_NAME = "your-project-name"  # <-- Needs to be set
VERSION = 1  # or whatever version you're using

try:
    workspace = rf.workspace(WORKSPACE)
    project = workspace.project(PROJECT_NAME)
    version = project.version(VERSION)
    
    print(f"Downloading {PROJECT_NAME} version {VERSION}...")
    version.download("yolov8", location="../dataset")
    print("Download complete!")
    
    # Verify download
    import os
    if os.path.exists("../dataset/data.yaml"):
        print("✓ data.yaml found!")
        with open("../dataset/data.yaml", "r") as f:
            print(f.read())
    else:
        print("✗ data.yaml still not found")
except Exception as e:
    print(f"Error: {e}")
    print("\nPlease check your workspace and project names")
