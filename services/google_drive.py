"""
Google Drive Integration Module
Handles downloading individual files from Google Drive shareable links
"""
import os
import re
import requests
from typing import Dict, Any, Optional

from config.config import Config


class GoogleDriveClient:
    """Client for downloading files from Google Drive shareable links"""
    
    def __init__(self):
        """Initialize Google Drive client"""
        self.cache_dir = Config.GDRIVE_CACHE_DIR
    
    def get_file_metadata(self, file_id: str) -> Dict[str, str]:
        """
        Get file metadata including name from Google Drive
        
        Args:
            file_id: Google Drive file ID
            
        Returns:
            Dictionary with file metadata (name, mimeType, etc.)
        """
        try:
            # Use Google Drive API v3 to get file metadata (no auth needed for public files)
            api_url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
            params = {
                'fields': 'name,mimeType,size,webViewLink',
                'supportsAllDrives': 'true'
            }
            
            response = requests.get(api_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'name': data.get('name', f'file_{file_id}'),
                    'mimeType': data.get('mimeType', 'application/octet-stream'),
                    'size': data.get('size', '0'),
                    'webViewLink': data.get('webViewLink', f"https://drive.google.com/file/d/{file_id}/view")
                }
            else:
                print(f"⚠️ Could not fetch metadata for file {file_id}, using default name")
                return {
                    'name': f'file_{file_id}',
                    'mimeType': 'application/octet-stream',
                    'size': '0',
                    'webViewLink': f"https://drive.google.com/file/d/{file_id}/view"
                }
                
        except Exception as e:
            print(f"Error fetching metadata: {str(e)}")
            return {
                'name': f'file_{file_id}',
                'mimeType': 'application/octet-stream',
                'size': '0',
                'webViewLink': f"https://drive.google.com/file/d/{file_id}/view"
            }
    
    def download_file_from_link(self, file_link: str) -> Optional[Dict[str, str]]:
        """
        Download a file from Google Drive link and return file info
        
        Args:
            file_link: Google Drive file URL
            
        Returns:
            Dictionary with file info (name, path, id, link) or None if failed
        """
        try:
            # Extract file ID
            file_id = self.extract_file_id_from_url(file_link)
            if not file_id:
                print(f"❌ Could not extract file ID from: {file_link}")
                return None
            
            # Get file metadata to get actual filename (may fail without API key)
            metadata = self.get_file_metadata(file_id)
            file_name = metadata['name']
            
            print(f"Downloading: {file_name} (ID: {file_id})")
            
            # Download the file
            try:
                # Create cache directory
                os.makedirs(self.cache_dir, exist_ok=True)
                
                # Use direct download URL for public files
                download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                
                # Start a session to handle cookies/redirects for large files
                session = requests.Session()
                response = session.get(download_url, stream=True, allow_redirects=True)
                
                # Check if we need to confirm download (for large files)
                if 'download_warning' in response.text or 'virus scan warning' in response.text:
                    # Extract confirmation token
                    for key, value in response.cookies.items():
                        if key.startswith('download_warning'):
                            download_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={value}"
                            response = session.get(download_url, stream=True, allow_redirects=True)
                            break
                
                if response.status_code == 200:
                    # Try to get filename from Content-Disposition header
                    content_disposition = response.headers.get('content-disposition', '')
                    if content_disposition and 'filename=' in content_disposition:
                        import re
                        filename_match = re.findall('filename="?([^"]+)"?', content_disposition)
                        if filename_match:
                            file_name = filename_match[0]
                    
                    # Download to temp file first
                    temp_path = os.path.join(self.cache_dir, f"temp_{file_id}")
                    
                    total_size = int(response.headers.get('content-length', 0))
                    with open(temp_path, 'wb') as f:
                        if total_size == 0:
                            f.write(response.content)
                        else:
                            downloaded = 0
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    downloaded += len(chunk)
                    
                    # Detect file type if filename is generic
                    if file_name.startswith('file_'):
                        detected_ext = self._detect_file_type(temp_path)
                        if detected_ext:
                            file_name = f"{file_name}.{detected_ext}"
                    
                    # Rename to final filename
                    file_path = os.path.join(self.cache_dir, file_name)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    os.rename(temp_path, file_path)
                    
                    print(f"✓ Downloaded: {file_name} ({os.path.getsize(file_path)} bytes)")
                    
                    return {
                        'name': file_name,
                        'path': file_path,
                        'id': file_id,
                        'link': file_link,
                        'source_type': 'gdrive'
                    }
                else:
                    print(f"❌ Failed to download {file_name}: HTTP {response.status_code}")
                    return None
                    
            except Exception as e:
                print(f"❌ Error downloading {file_name}: {str(e)}")
                return None
            
        except Exception as e:
            print(f"❌ Error downloading from link: {str(e)}")
            return None
    
    def _detect_file_type(self, file_path: str) -> Optional[str]:
        """
        Detect file type from content
        
        Args:
            file_path: Path to file
            
        Returns:
            File extension or None
        """
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)
            
            # PDF
            if header.startswith(b'%PDF'):
                return 'pdf'
            # DOCX (ZIP-based)
            elif header.startswith(b'PK\x03\x04'):
                # Could be DOCX, XLSX, or other ZIP
                with open(file_path, 'rb') as f:
                    content = f.read(512)
                    if b'word/' in content:
                        return 'docx'
                    elif b'xl/' in content:
                        return 'xlsx'
                return 'zip'
            # PNG
            elif header.startswith(b'\x89PNG'):
                return 'png'
            # JPEG
            elif header.startswith(b'\xff\xd8\xff'):
                return 'jpg'
            # Plain text
            elif all(b >= 32 or b in (9, 10, 13) for b in header[:100]):
                return 'txt'
            
            return None
        except Exception as e:
            print(f"Error detecting file type: {str(e)}")
            return None
    
    def extract_file_id_from_url(self, url: str) -> Optional[str]:
        """
        Extract file ID from Google Drive URL
        
        Args:
            url: Google Drive file URL
            
        Returns:
            File ID or None
        """
        patterns = [
            r'/d/([a-zA-Z0-9_-]+)',
            r'id=([a-zA-Z0-9_-]+)',
            r'/file/d/([a-zA-Z0-9_-]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
