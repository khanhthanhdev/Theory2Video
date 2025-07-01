# llm_config/notifications.py
"""
Notification service for user feedback.
"""

import gradio as gr
import logging
from typing import Optional
from .interfaces import INotificationService

logger = logging.getLogger(__name__)


class GradioNotificationService(INotificationService):
    """Gradio-specific notification service for user feedback."""
    
    def __init__(self):
        self.last_message: Optional[str] = None
        self.last_type: Optional[str] = None
    
    def show_success(self, message: str) -> None:
        """Show success notification."""
        self.last_message = f"✅ {message}"
        self.last_type = "success"
        logger.info(f"Success notification: {message}")
        # In a real Gradio app, you might want to trigger a UI update here
    
    def show_error(self, message: str) -> None:
        """Show error notification."""
        self.last_message = f"❌ {message}"
        self.last_type = "error"
        logger.error(f"Error notification: {message}")
    
    def show_warning(self, message: str) -> None:
        """Show warning notification."""
        self.last_message = f"⚠️ {message}"
        self.last_type = "warning"
        logger.warning(f"Warning notification: {message}")
    
    def show_info(self, message: str) -> None:
        """Show info notification."""
        self.last_message = f"ℹ️ {message}"
        self.last_type = "info"
        logger.info(f"Info notification: {message}")
    
    def get_last_notification(self) -> Optional[str]:
        """Get the last notification message."""
        return self.last_message
    
    def get_last_type(self) -> Optional[str]:
        """Get the type of the last notification."""
        return self.last_type
    
    def clear_notifications(self) -> None:
        """Clear all notifications."""
        self.last_message = None
        self.last_type = None
    
    def create_gradio_update(self, message: str, notification_type: str = "info") -> dict:
        """Create a Gradio update dict for displaying notifications."""
        icons = {
            "success": "✅",
            "error": "❌",
            "warning": "⚠️",
            "info": "ℹ️"
        }
        
        icon = icons.get(notification_type, "ℹ️")
        formatted_message = f"{icon} {message}"
        
        # Store the notification
        self.last_message = formatted_message
        self.last_type = notification_type
        
        return gr.update(
            value=formatted_message,
            visible=True
        )
    
    def format_validation_message(self, is_valid: bool, message: str) -> str:
        """Format a validation message with appropriate styling."""
        if is_valid:
            return f"✅ {message}"
        else:
            return f"❌ {message}"
    
    def format_status_message(self, status: str, details: str = "") -> str:
        """Format a status message."""
        if details:
            return f"📊 {status}: {details}"
        else:
            return f"📊 {status}"


class ConsoleNotificationService(INotificationService):
    """Console-based notification service for testing/development."""
    
    def show_success(self, message: str) -> None:
        """Show success notification."""
        print(f"✅ SUCCESS: {message}")
        logger.info(f"Success: {message}")
    
    def show_error(self, message: str) -> None:
        """Show error notification."""
        print(f"❌ ERROR: {message}")
        logger.error(f"Error: {message}")
    
    def show_warning(self, message: str) -> None:
        """Show warning notification."""
        print(f"⚠️ WARNING: {message}")
        logger.warning(f"Warning: {message}")
    
    def show_info(self, message: str) -> None:
        """Show info notification."""
        print(f"ℹ️ INFO: {message}")
        logger.info(f"Info: {message}")
