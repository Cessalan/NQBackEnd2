import subprocess
import sys

# The list of packages to uninstall
packages = [
    "fastapi",
    "uvicorn",
    "python-dotenv",
    "requests",
    "pydantic",
    "firebase-admin",
    "langchain",
    "langchain-community",
    "langchain-openai",
    "openai",
    "numpy==1.23.5",
    "faiss-cpu==1.7.4",
    "pypdf",
    "pandas",
    "tiktoken",
    "langchain-core",
    "langchain-text-splitters",
    "python-pptx",
    "docx2txt",
    "python-docx",
    "unstructured",
    "nltk",
    "openpyxl",
    "xlrd"
]

def uninstall_packages(packages_list):
    """
    Uninstalls a list of Python packages using pip.
    """
    print("Starting uninstallation process...")
    for package in packages_list:
        try:
            # The subprocess.check_call will raise an exception if the command fails.
            # The '--yes' flag automatically confirms the uninstallation.
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package])
            print(f"Successfully uninstalled {package}.")
        except subprocess.CalledProcessError as e:
            # This handles cases where a package might not be found.
            print(f"Failed to uninstall {package}. It might not be installed. Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while uninstalling {package}: {e}")

if __name__ == "__main__":
    uninstall_packages(packages)
    print("\nUninstallation process complete.")
    print("You can now safely create a new virtual environment and install your dependencies there.")
