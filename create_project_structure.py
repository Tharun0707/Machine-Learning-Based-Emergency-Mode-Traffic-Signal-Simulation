import os
import shutil

def create_project_structure():
    """Create the proper project structure"""
    
    print("🏗️ Creating project structure...")
    print("=" * 40)
    
    # Create main directories
    directories = [
        "backend",
        "backend/models",
        "backend/uploads",
        "backend/temp"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"📁 Directory already exists: {directory}")
    
    print("\n📋 Project structure created:")
    print("├── backend/")
    print("│   ├── models/          # Place your best.pt here")
    print("│   ├── uploads/         # Temporary file uploads")
    print("│   ├── temp/           # Temporary processing files")
    print("│   ├── model_server.py # Main server file")
    print("│   └── requirements.txt # Python dependencies")
    print("└── (your Next.js files)")
    
    print("\n🎯 Next steps:")
    print("1. Copy your 'best.pt' file to the 'backend/models/' folder")
    print("2. Navigate to the backend folder: cd backend")
    print("3. Run the setup script")

if __name__ == "__main__":
    create_project_structure()
