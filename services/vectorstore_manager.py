"""
VectorStore Manager
Handles loading, saving, and managing vectorstores in Firebase Storage.
"""

import os
import asyncio
import shutil
import re
from typing import Optional, List, Dict, Tuple
from uuid import uuid4

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from firebase_admin import storage

import tempfile
import platform


class VectorStoreManager:
    """
    Manages FAISS vectorstores for chat sessions.
    
    Handles:
    - Loading existing vectorstores from Firebase
    - Saving combined and per-file vectorstores to Firebase
    - Caching vectorstores in memory
    """
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        
    @staticmethod
    def get_temp_dir() -> str:
        """Get appropriate temp directory for the current environment."""
        if platform.system() == "Linux":
            return "/tmp"
        else:
            return tempfile.gettempdir()

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename for Windows filesystem compatibility.
        Removes/replaces special characters that cause issues with temp paths.
        """
        # Remove accents and special characters
        # Keep only alphanumeric, dots, dashes, and underscores
        sanitized = re.sub(r'[^\w\s.-]', '', filename)
        # Replace spaces with underscores
        sanitized = sanitized.replace(' ', '_')
        # Remove any consecutive dots or dashes
        sanitized = re.sub(r'[.-]{2,}', '_', sanitized)
        return sanitized
    
    # ========================================
    # LOAD VECTORSTORES
    # ========================================
    
    async def load_combined_vectorstore_from_firebase(self, chat_id: str) -> Optional[FAISS]:
        """
        Load the combined vectorstore from Firebase Storage.
        
        Location: vectorstores/{chat_id}/
        
        Args:
            chat_id: The chat session ID
            
        Returns:
            FAISS vectorstore if found, None otherwise
        """
        try:
            print(f"🔍 Checking for combined vectorstore: vectorstores/{chat_id}/")
            
            bucket = storage.bucket()
            prefix = f"vectorstores/{chat_id}/"
            
            # Check if vectorstore exists
            blobs = list(bucket.list_blobs(prefix=prefix))
            
            if not blobs:
                print(f"📭 No combined vectorstore found for {chat_id}")
                return None
            
            print(f"📥 Downloading combined vectorstore ({len(blobs)} files)...")
            
            # Create temp directory
            temp_dir = os.path.join(self.get_temp_dir(), f"vectorstore_{chat_id}_{uuid4().hex[:8]}")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Download all files
            for blob in blobs:
                relative_path = blob.name.replace(prefix, "")
                if not relative_path:  # Skip if it's just the folder
                    continue
                
                local_path = os.path.join(temp_dir, relative_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                blob.download_to_filename(local_path)
                print(f"  ✅ Downloaded {relative_path}")
            
            # Load vectorstore
            print(f"🔤 Loading vectorstore into memory...")
            vectorstore = FAISS.load_local(
                temp_dir,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Cleanup temp files
            shutil.rmtree(temp_dir)
            print(f"✅ Loaded combined vectorstore, cleaned up temp files")
            
            return vectorstore
            
        except Exception as e:
            print(f"⚠️ Could not load combined vectorstore for {chat_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # ========================================
# BATCH UPLOAD
# ========================================
    
    async def upload_all_vectorstores(
    self, 
    chat_id: str, 
    combined_vectorstore: FAISS,
    file_documents: Dict[str, List[Document]]
) -> Dict[str, any]:
        """
        Upload both combined and per-file vectorstores in parallel.
        
        Args:
            chat_id: The chat session ID
            combined_vectorstore: The combined vectorstore
            file_documents: Dict mapping filename -> list of documents
            
        Returns:
            Dict with upload results and any warnings
        """
        print(f"🚀 Starting parallel vectorstore uploads...")
        
        # Prepare tasks
        tasks = []
        task_names = []
        
        # Combined vectorstore task
        tasks.append(self.upload_combined_vectorstore_to_firebase(chat_id, combined_vectorstore))
        task_names.append("combined")
        
        # Per-file vectorstore tasks
        for filename, documents in file_documents.items():
            tasks.append(self.upload_file_vectorstore_to_firebase(chat_id, filename, documents))
            task_names.append(f"file:{filename}")
        
        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        warnings = []
        combined_success = False
        
        for i, (task_name, result) in enumerate(zip(task_names, results)):
            if isinstance(result, Exception):
                error_msg = str(result)
                print(f"❌ Task {task_name} failed with exception: {error_msg}")
                warnings.append({
                    "task": task_name,
                    "error": error_msg
                })
            else:
                success, error_msg = result
                if task_name == "combined":
                    combined_success = success
                    if not success:
                        print(f"❌ Combined vectorstore upload failed: {error_msg}")
                else:
                    if not success:
                        filename = task_name.replace("file:", "")
                        print(f"⚠️ File vectorstore upload failed for {filename}: {error_msg}")
                        warnings.append({
                            "filename": filename,
                            "error": error_msg
                        })
        
        print(f"✅ Parallel uploads complete. Combined: {combined_success}, Warnings: {len(warnings)}")
        
        return {
            "combined_success": combined_success,
            "warnings": warnings,
            "total_tasks": len(tasks),
            "failed_tasks": len(warnings)
        }

    
    async def load_file_vectorstore_from_firebase(
        self, 
        chat_id: str, 
        filename: str
    ) -> Optional[FAISS]:
        """
        Load a per-file vectorstore from Firebase Storage.
        
        Location: FileVectorStore/{chat_id}/{filename}/
        
        Args:
            chat_id: The chat session ID
            filename: The specific file name
            
        Returns:
            FAISS vectorstore if found, None otherwise
        """
        try:
            print(f"🔍 Loading file vectorstore: FileVectorStore/{chat_id}/{filename}/")
            
            bucket = storage.bucket()
            prefix = f"FileVectorStore/{chat_id}/{filename}/"
            
            # Check if vectorstore exists
            blobs = list(bucket.list_blobs(prefix=prefix))
            
            if not blobs:
                print(f"📭 No vectorstore found for {filename}")
                return None
            
            # Create temp directory
            temp_dir = os.path.join(self.get_temp_dir(), f"file_vs_{chat_id}_{uuid4().hex[:8]}")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Download all files
            for blob in blobs:
                relative_path = blob.name.replace(prefix, "")
                if not relative_path:
                    continue
                
                local_path = os.path.join(temp_dir, relative_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                blob.download_to_filename(local_path)
            
            # Load vectorstore
            vectorstore = FAISS.load_local(
                temp_dir,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Cleanup
            shutil.rmtree(temp_dir)
            print(f"✅ Loaded vectorstore for {filename}")
            
            return vectorstore
            
        except Exception as e:
            print(f"⚠️ Could not load file vectorstore for {filename}: {e}")
            return None

    # ========================================
    # SAVE VECTORSTORES
    # ========================================
        
    async def upload_combined_vectorstore_to_firebase(
            self, 
            chat_id: str, 
            vectorstore: FAISS
        ) -> Tuple[bool, Optional[str]]:
            """
            Upload combined vectorstore to Firebase Storage.
            
            Location: vectorstores/{chat_id}/
            
            Args:
                chat_id: The chat session ID
                vectorstore: The FAISS vectorstore to upload
                
            Returns:
                Tuple of (success: bool, error_message: Optional[str])
            """
            temp_dir = None
            try:
                print(f"☁️ Uploading combined vectorstore for {chat_id}...")
                
                # Save locally first
                temp_dir = os.path.join(self.get_temp_dir(), f"vectorstore_{chat_id}_{uuid4().hex[:8]}")
                os.makedirs(temp_dir, exist_ok=True)
                
                vectorstore.save_local(temp_dir)
                print(f"💾 Saved vectorstore locally to {temp_dir}")
                
                # Upload to Firebase
                bucket = storage.bucket()
                uploaded_files = []
                
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        local_path = os.path.join(root, file)
                        relative_path = os.path.relpath(local_path, temp_dir)
                        
                        # Firebase path
                        firebase_path = f"vectorstores/{chat_id}/{relative_path}"
                        
                        blob = bucket.blob(firebase_path)
                        blob.upload_from_filename(local_path)
                        uploaded_files.append(relative_path)
                        print(f"  ✅ Uploaded {relative_path}")
                
                print(f"✅ Combined vectorstore uploaded ({len(uploaded_files)} files)")
                
                return True, None
                
            except Exception as e:
                error_msg = f"Failed to upload combined vectorstore: {str(e)}"
                print(f"❌ {error_msg}")
                import traceback
                traceback.print_exc()
                return False, error_msg
                
            finally:
                # Cleanup temp directory
                if temp_dir and os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir)
                        print(f"🗑️ Cleaned up temp directory")
                    except Exception as cleanup_error:
                        print(f"⚠️ Failed to cleanup {temp_dir}: {cleanup_error}")
        
    async def upload_file_vectorstore_to_firebase(
        self, 
        chat_id: str, 
        filename: str, 
        documents: List[Document]
    ) -> Tuple[bool, Optional[str]]:
        """
        Create and upload a per-file vectorstore to Firebase Storage.
        
        Location: FileVectorStore/{chat_id}/{filename}/
        
        Args:
            chat_id: The chat session ID
            filename: The file name
            documents: List of Document objects for this file
            
        Returns:
            Tuple of (success: bool, error_message: Optional[str])
        """
        temp_dir = None
        try:
            print(f"☁️ Uploading file vectorstore for {filename}...")
            print(f"   Documents to process: {len(documents)}")
            
            # Check if documents list is empty
            if not documents:
                error_msg = f"No documents provided for {filename}"
                print(f"⚠️ {error_msg}")
                return False, error_msg
            
            # Create vectorstore from documents
            print(f"🔤 Creating vectorstore from {len(documents)} documents...")
            file_vectorstore = FAISS.from_documents(documents, self.embeddings)
            print(f"✅ Vectorstore created successfully")

            # Save locally first
            safe_filename = self.sanitize_filename(filename)
            temp_dir = os.path.join(
                self.get_temp_dir(),
                f"file_vs_{chat_id}_{safe_filename}_{uuid4().hex[:8]}"
            )
            os.makedirs(temp_dir, exist_ok=True)
            print(f"💾 Saving to temp directory: {temp_dir}")
            print(f"   Original filename: {filename}")
            print(f"   Sanitized filename: {safe_filename}")
            
            file_vectorstore.save_local(temp_dir)
            print(f"✅ Saved file vectorstore locally")
            
            # List files in temp directory
            local_files = os.listdir(temp_dir)
            print(f"📁 Files in temp dir: {local_files}")
            
            # Upload to Firebase
            bucket = storage.bucket()
            uploaded_files = []
            
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, temp_dir)
                    
                    # Firebase path
                    firebase_path = f"FileVectorStore/{chat_id}/{filename}/{relative_path}"
                    
                    print(f"📤 Uploading: {local_path}")
                    print(f"   → Firebase path: {firebase_path}")
                    
                    blob = bucket.blob(firebase_path)
                    blob.upload_from_filename(local_path)
                    uploaded_files.append(relative_path)
                    print(f"  ✅ Uploaded {relative_path}")
            
            print(f"✅ File vectorstore uploaded for {filename} ({len(uploaded_files)} files)")
            
            return True, None
            
        except Exception as e:
            error_msg = f"Failed to upload file vectorstore for {filename}: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return False, error_msg
            
        finally:
            # Cleanup temp directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    print(f"🗑️ Cleaned up temp directory: {temp_dir}")
                except Exception as cleanup_error:
                    print(f"⚠️ Failed to cleanup {temp_dir}: {cleanup_error}")
        

# Global instance
vectorstore_manager = VectorStoreManager()