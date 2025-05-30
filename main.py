import os
import re
import logging
import asyncio
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import uuid
import shutil

# FastAPI imports for API endpoints
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MemberDirectoryModel:
    """Data model for member directory information"""
    # Personal Information
    first_name: str
    last_name: str
    birthday: Optional[str] = None

    # Contact Information
    phone: Optional[str] = None
    mobile: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None

    # Emergency Contact
    emergency_name: Optional[str] = None
    emergency_relation: Optional[str] = None
    emergency_phone: Optional[str] = None

    # Privacy Settings
    phone_visible: bool = False
    mobile_visible: bool = False
    email_visible: bool = False
    address_visible: bool = False
    birthday_visible: bool = False
    emergency_visible: bool = False

    # Consent
    consent_given: bool = False


class MemberDirectoryRequest(BaseModel):
    """Pydantic model for API request validation"""
    # Personal Information
    first_name: str
    last_name: str
    birthday: Optional[str] = None

    # Contact Information
    phone: Optional[str] = None
    mobile: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None

    # Emergency Contact
    emergency_name: Optional[str] = None
    emergency_relation: Optional[str] = None
    emergency_phone: Optional[str] = None

    # Privacy Settings
    phone_visible: bool = False
    mobile_visible: bool = False
    email_visible: bool = False
    address_visible: bool = False
    birthday_visible: bool = False
    emergency_visible: bool = False

    # Consent
    consent_given: bool = False

    @validator('first_name', 'last_name')
    def validate_required_names(cls, v):
        if not v or not v.strip():
            raise ValueError('First name and last name are required')
        return v.strip()

    @validator('phone', 'mobile', 'emergency_phone')
    def validate_phone_format(cls, v):
        if v and not re.match(r'^[\d\s\-\(\)\+\.]+$', v):
            raise ValueError('Invalid phone number format')
        return v


class MemberDirectoryService:
    """Business logic service for managing member directory operations"""

    def __init__(self, data_storage_path: Optional[str] = None):
        """
        Initialize the member directory service

        Args:
            data_storage_path: Custom path for storing member files
        """
        self.data_storage_path = Path(data_storage_path) if data_storage_path else Path.cwd() / "MemberDirectoryData"

        # Ensure the directory exists
        self.data_storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Member directory service initialized with path: {self.data_storage_path}")

    def clean_phone_number(self, phone_number: str) -> str:
        """
        Clean phone number for use as filename

        Args:
            phone_number: Raw phone number string

        Returns:
            Cleaned phone number suitable for filename
        """
        if not phone_number or not phone_number.strip():
            return "unknown"

        # Remove all non-digit characters
        cleaned = re.sub(r'[^\d]', '', phone_number)

        # Ensure we have a valid length
        if len(cleaned) >= 10:
            return cleaned

        # If phone number is too short, pad with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{cleaned}_{timestamp}"

    def generate_member_directory_text(self, member_data: MemberDirectoryModel) -> str:
        """
        Generate formatted text content for the member directory file

        Args:
            member_data: Member information

        Returns:
            Formatted text content
        """
        lines = []

        lines.extend([
            "===========================================",
            "      ANAND VIHAR MEMBER DIRECTORY",
            "===========================================",
            "",
            f"Record Created: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
            f"Record ID: {str(uuid.uuid4())}",
            "",
        ])

        lines.extend([
            "PERSONAL INFORMATION:",
            "-------------------------------------------",
            f"Name: {member_data.first_name} {member_data.last_name}",
        ])

        if member_data.birthday:
            lines.append(f"Birthday: {member_data.birthday}")
        lines.append("")

        lines.extend([
            "CONTACT INFORMATION:",
            "-------------------------------------------",
        ])

        if member_data.phone:
            lines.append(f"Home Phone: {member_data.phone}")
        if member_data.mobile:
            lines.append(f"Mobile: {member_data.mobile}")
        if member_data.email:
            lines.append(f"Email: {member_data.email}")
        if member_data.address:
            lines.append(f"Address: {member_data.address}")
        lines.append("")

        lines.extend([
            "EMERGENCY CONTACT:",
            "-------------------------------------------",
        ])

        if member_data.emergency_name:
            lines.append(f"Name: {member_data.emergency_name}")
        if member_data.emergency_relation:
            lines.append(f"Relationship: {member_data.emergency_relation}")
        if member_data.emergency_phone:
            lines.append(f"Phone: {member_data.emergency_phone}")
        lines.append("")

        lines.extend([
            "PRIVACY SETTINGS:",
            "-------------------------------------------",
            f"Home Phone Visible: {'Yes' if member_data.phone_visible else 'No'}",
            f"Mobile Visible: {'Yes' if member_data.mobile_visible else 'No'}",
            f"Email Visible: {'Yes' if member_data.email_visible else 'No'}",
            f"Address Visible: {'Yes' if member_data.address_visible else 'No'}",
            f"Birthday Visible: {'Yes' if member_data.birthday_visible else 'No'}",
            f"Emergency Contact Visible: {'Yes' if member_data.emergency_visible else 'No'}",
            "",
            f"Consent Given: {'Yes' if member_data.consent_given else 'No'}",
            "",
        ])

        phone_for_filename = member_data.phone if member_data.phone else member_data.mobile
        clean_phone = self.clean_phone_number(phone_for_filename)

        lines.extend([
            "===========================================",
            f"File saved as: {clean_phone}.txt",
            "===========================================",
        ])

        return "\n".join(lines)

    async def save_member_directory_async(self, member_data: MemberDirectoryModel) -> Tuple[bool, str, Optional[str]]:
        """
        Save member directory information to a text file

        Args:
            member_data: Member information from the form

        Returns:
            Tuple of (success, message, filename)
        """
        try:
            # Validate required fields
            if not member_data.first_name or not member_data.last_name:
                return False, "First Name and Last Name are required.", None

            # Use home phone as primary, fallback to mobile if home phone is empty
            phone_for_filename = member_data.phone if member_data.phone else member_data.mobile

            if not phone_for_filename:
                return False, "At least one phone number (Home or Mobile) is required.", None

            # Clean phone number for filename
            clean_phone = self.clean_phone_number(phone_for_filename)
            file_name = f"{clean_phone}.txt"
            full_path = self.data_storage_path / file_name

            # Check if file already exists and create backup
            if full_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file_name = f"{clean_phone}_backup_{timestamp}.txt"
                backup_path = self.data_storage_path / backup_file_name
                shutil.copy2(full_path, backup_path)
                logger.info(f"Existing member file backed up: {backup_file_name}")

            # Generate the text content
            file_content = self.generate_member_directory_text(member_data)

            # Save to file asynchronously
            await asyncio.to_thread(full_path.write_text, file_content, encoding='utf-8')

            logger.info(f"Member directory saved successfully: {file_name}")
            return True, "Member directory saved successfully.", file_name

        except PermissionError as e:
            logger.error(f"Permission denied when saving member directory: {e}")
            return False, "Access denied. Please check file permissions.", None
        except FileNotFoundError as e:
            logger.error(f"Directory not found when saving member directory: {e}")
            return False, "Storage directory not found.", None
        except OSError as e:
            logger.error(f"OS error when saving member directory: {e}")
            return False, "File system error occurred.", None
        except Exception as e:
            logger.error(f"Unexpected error when saving member directory: {e}")
            return False, "An unexpected error occurred.", None

    async def get_member_by_phone_async(self, phone_number: str) -> Tuple[bool, Optional[str], str]:
        """
        Retrieve member information by phone number

        Args:
            phone_number: Phone number to search for

        Returns:
            Tuple of (success, content, message)
        """
        try:
            clean_phone = self.clean_phone_number(phone_number)
            file_name = f"{clean_phone}.txt"
            full_path = self.data_storage_path / file_name

            if not full_path.exists():
                return False, None, "Member not found."

            content = await asyncio.to_thread(full_path.read_text, encoding='utf-8')
            return True, content, f"Member file found: {file_name}"

        except Exception as e:
            logger.error(f"Error retrieving member by phone {phone_number}: {e}")
            return False, None, "Error retrieving member information."

    def list_all_members(self) -> Tuple[bool, Optional[List[str]], str]:
        """
        List all member directory files

        Returns:
            Tuple of (success, file_names, message)
        """
        try:
            if not self.data_storage_path.exists():
                return False, None, "Member directory not found."

            # Get all .txt files excluding backup files
            files = [
                f.name for f in self.data_storage_path.glob("*.txt")
                if "_backup_" not in f.name
            ]

            return True, files, f"Found {len(files)} member records."

        except Exception as e:
            logger.error(f"Error listing member files: {e}")
            return False, None, "Error retrieving member list."


# FastAPI application
app = FastAPI(title="Anand Vihar Member Directory API", version="1.0.0")

# Initialize service
member_service = MemberDirectoryService()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://anand-vihar.onrender.com"],  # Or ["http://localhost:3000"] if React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/member-directory/save")
async def save_member_directory(member_data: MemberDirectoryRequest):
    """
    Save member directory information

    Args:
        member_data: Member information from the form

    Returns:
        JSON response with success status
    """
    try:
        # Convert Pydantic model to dataclass
        member_model = MemberDirectoryModel(
            first_name=member_data.first_name,
            last_name=member_data.last_name,
            birthday=member_data.birthday,
            phone=member_data.phone,
            mobile=member_data.mobile,
            email=member_data.email,
            address=member_data.address,
            emergency_name=member_data.emergency_name,
            emergency_relation=member_data.emergency_relation,
            emergency_phone=member_data.emergency_phone,
            phone_visible=member_data.phone_visible,
            mobile_visible=member_data.mobile_visible,
            email_visible=member_data.email_visible,
            address_visible=member_data.address_visible,
            birthday_visible=member_data.birthday_visible,
            emergency_visible=member_data.emergency_visible,
            consent_given=member_data.consent_given
        )

        success, message, file_name = await member_service.save_member_directory_async(member_model)

        if success:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "success": True,
                    "message": message,
                    "file_name": file_name
                }
            )
        else:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "message": message
                }
            )

    except Exception as e:
        logger.error(f"Error in save_member_directory API: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error occurred."
        )


@app.get("/api/member-directory/list")
async def list_members():
    """
    List all member directory files

    Returns:
        JSON response with list of member files
    """
    try:
        success, files, message = member_service.list_all_members()

        if success:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "success": True,
                    "message": message,
                    "files": files
                }
            )
        else:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "success": False,
                    "message": message
                }
            )

    except Exception as e:
        logger.error(f"Error in list_members API: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error occurred."
        )


@app.get("/api/member-directory/member/{phone_number}")
async def get_member(phone_number: str):
    """
    Get member information by phone number

    Args:
        phone_number: Phone number to search for

    Returns:
        JSON response with member information
    """
    try:
        success, content, message = await member_service.get_member_by_phone_async(phone_number)

        if success:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "success": True,
                    "message": message,
                    "content": content
                }
            )
        else:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "success": False,
                    "message": message
                }
            )

    except Exception as e:
        logger.error(f"Error in get_member API: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error occurred."
        )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Member Directory API"}


# Example usage and testing
if __name__ == "__main__":
    import uvicorn


    # Example of how to use the service directly
    async def test_service():
        """Test function to demonstrate service usage"""
        service = MemberDirectoryService()

        # Create test member data
        test_member = MemberDirectoryModel(
            first_name="John",
            last_name="Doe",
            birthday="1950-01-15",
            phone="(813) 555-1234",
            mobile="(813) 555-5678",
            email="john.doe@example.com",
            address="1115 Narmada Way, Wesley Chapel, FL 33543",
            emergency_name="Jane Doe",
            emergency_relation="spouse",
            emergency_phone="(813) 555-9999",
            phone_visible=True,
            mobile_visible=True,
            email_visible=False,
            address_visible=True,
            birthday_visible=True,
            emergency_visible=False,
            consent_given=True
        )

        # Save the member
        success, message, filename = await service.save_member_directory_async(test_member)
        print(f"Save result: {success}, {message}, {filename}")

        # List all members
        success, files, message = service.list_all_members()
        print(f"List result: {success}, {message}, Files: {files}")


    # Run test
    # asyncio.run(test_service())

    # Start the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
