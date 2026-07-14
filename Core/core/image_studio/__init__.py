"""
Image_Studio — image generation and editing subsystem.

Generates and edits images via the Model Provider (image capability),
maintains a session image history for edit-target resolution, saves
images to a designated user-accessible folder, and reports failures
clearly.

Design: Image_Studio.
Requirements: 15.1–15.6.
"""

from .image_studio import ImageEntry, ImageStudio, ImageResult, GenerationFailure

__all__ = ["ImageStudio", "ImageEntry", "ImageResult", "GenerationFailure"]
