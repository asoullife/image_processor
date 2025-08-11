"""Thai language translations for the human review system."""

from typing import Dict, List

class ThaiTranslations:
    """Thai language translations for rejection reasons and UI elements."""
    
    # Rejection reason translations
    REJECTION_REASONS = {
        "low_quality": "คุณภาพภาพต่ำ ไม่เหมาะสำหรับขาย",
        "defects_detected": "พบความผิดปกติในภาพ",
        "similar_images": "พบภาพที่คล้ายกัน อาจถูกปฏิเสธเป็น spam",
        "compliance_issues": "ปัญหาลิขสิทธิ์หรือความเป็นส่วนตัว",
        "technical_issues": "ปัญหาทางเทคนิค",
        "metadata_issues": "ปัญหาข้อมูลเมตาดาต้า",
        "inappropriate_content": "เนื้อหาไม่เหมาะสม",
        "copyright_violation": "ละเมิดลิขสิทธิ์",
        "privacy_violation": "ละเมิดความเป็นส่วนตัว",
        "trademark_detected": "พบเครื่องหมายการค้า",
        "face_detected": "พบใบหน้าในภาพ",
        "license_plate_detected": "พบป้ายทะเบียนรถ",
        "logo_detected": "พบโลโก้หรือเครื่องหมายการค้า",
        "text_overlay": "พบข้อความซ้อนทับในภาพ",
        "watermark_detected": "พบลายน้ำในภาพ",
        "poor_composition": "องค์ประกอบภาพไม่ดี",
        "overexposed": "ภาพสว่างเกินไป",
        "underexposed": "ภาพมืดเกินไป",
        "blurry": "ภาพเบลอ",
        "noisy": "ภาพมีสัญญาณรบกวน",
        "low_resolution": "ความละเอียดต่ำ",
        "aspect_ratio_issues": "อัตราส่วนภาพไม่เหมาะสม",
        "color_issues": "ปัญหาสีในภาพ",
        "duplicate_content": "เนื้อหาซ้ำกับภาพอื่น",
        "editorial_content": "เนื้อหาเชิงข่าว ไม่เหมาะสำหรับ stock",
        "model_release_required": "ต้องมี model release",
        "property_release_required": "ต้องมี property release"
    }
    
    # Detailed explanations for rejection reasons
    REJECTION_EXPLANATIONS = {
        "low_quality": "ภาพนี้มีคุณภาพต่ำ อาจเป็นเพราะความคมชัดไม่เพียงพอ หรือมีสัญญาณรบกวนมาก ทำให้ไม่เหมาะสำหรับการขายใน Adobe Stock",
        "defects_detected": "พบความผิดปกติในภาพ เช่น จุดด่าง รอยขีดข่วน หรือวัตถุแปลกปลอมที่ไม่ควรอยู่ในภาพ",
        "similar_images": "พบภาพที่คล้ายกันมาก อาจถูก Adobe Stock ปฏิเสธเป็น spam หรือเนื้อหาซ้ำ ควรเลือกภาพที่ดีที่สุดเพียงภาพเดียว",
        "compliance_issues": "ภาพนี้อาจมีปัญหาเรื่องลิขสิทธิ์หรือความเป็นส่วนตัว เช่น พบใบหน้า โลโก้ หรือเครื่องหมายการค้า",
        "technical_issues": "พบปัญหาทางเทคนิค เช่น ความละเอียดต่ำ อัตราส่วนไม่เหมาะสม หรือรูปแบบไฟล์ไม่ถูกต้อง",
        "face_detected": "พบใบหน้าในภาพ ต้องมี model release หรือใบหน้าต้องไม่สามารถระบุตัวตนได้",
        "logo_detected": "พบโลโก้หรือเครื่องหมายการค้าในภาพ ซึ่งอาจละเมิดลิขสิทธิ์",
        "duplicate_content": "เนื้อหาซ้ำกับภาพอื่นที่มีอยู่แล้ว ควรเลือกภาพที่มีความหลากหลายมากกว่า"
    }
    
    # UI element translations
    UI_ELEMENTS = {
        "approve": "อนุมัติ",
        "reject": "ปฏิเสธ",
        "approve_all": "อนุมัติทั้งหมด",
        "reject_all": "ปฏิเสธทั้งหมด",
        "review_completed": "ตรวจสอบเรียบร้อย",
        "pending_review": "รอตรวจสอบ",
        "human_reviewed": "ตรวจสอบโดยมนุษย์แล้ว",
        "ai_decision": "การตัดสินใจของ AI",
        "similarity_score": "คะแนนความคล้าย",
        "quality_score": "คะแนนคุณภาพ",
        "confidence_score": "คะแนนความมั่นใจ",
        "processing_time": "เวลาประมวลผล",
        "source_folder": "โฟลเดอร์ต้นทาง",
        "output_folder": "โฟลเดอร์ผลลัพธ์",
        "filter_by": "กรองตาม",
        "sort_by": "เรียงตาม",
        "search": "ค้นหา",
        "select_all": "เลือกทั้งหมด",
        "clear_selection": "ยกเลิกการเลือก",
        "bulk_actions": "การดำเนินการแบบกลุ่ม",
        "view_similar": "ดูภาพที่คล้าย",
        "compare_images": "เปรียบเทียบภาพ",
        "zoom_in": "ขยายภาพ",
        "zoom_out": "ย่อภาพ",
        "full_screen": "เต็มจอ",
        "previous_image": "ภาพก่อนหน้า",
        "next_image": "ภาพถัดไป",
        "image_details": "รายละเอียดภาพ",
        "processing_results": "ผลการประมวลผล",
        "review_history": "ประวัติการตรวจสอบ"
    }
    
    # Status messages
    STATUS_MESSAGES = {
        "image_approved": "ภาพได้รับการอนุมัติแล้ว",
        "image_rejected": "ภาพถูกปฏิเสธแล้ว",
        "bulk_approve_success": "อนุมัติภาพทั้งหมดเรียบร้อยแล้ว",
        "bulk_reject_success": "ปฏิเสธภาพทั้งหมดเรียบร้อยแล้ว",
        "review_saved": "บันทึกการตรวจสอบแล้ว",
        "error_occurred": "เกิดข้อผิดพลาด",
        "loading": "กำลังโหลด...",
        "no_images_found": "ไม่พบภาพ",
        "no_similar_images": "ไม่พบภาพที่คล้าย",
        "processing_complete": "ประมวลผลเสร็จสิ้น",
        "review_required": "ต้องตรวจสอบ",
        "auto_approved": "อนุมัติอัตโนมัติ",
        "auto_rejected": "ปฏิเสธอัตโนมัติ"
    }
    
    @classmethod
    def get_rejection_reason(cls, reason_key: str) -> str:
        """Get Thai translation for rejection reason."""
        return cls.REJECTION_REASONS.get(reason_key, reason_key)
    
    @classmethod
    def get_rejection_explanation(cls, reason_key: str) -> str:
        """Get detailed Thai explanation for rejection reason."""
        return cls.REJECTION_EXPLANATIONS.get(reason_key, cls.REJECTION_REASONS.get(reason_key, reason_key))
    
    @classmethod
    def get_ui_text(cls, key: str) -> str:
        """Get Thai translation for UI element."""
        return cls.UI_ELEMENTS.get(key, key)
    
    @classmethod
    def get_status_message(cls, key: str) -> str:
        """Get Thai status message."""
        return cls.STATUS_MESSAGES.get(key, key)
    
    @classmethod
    def translate_rejection_reasons(cls, reasons: List[str]) -> List[Dict[str, str]]:
        """Translate a list of rejection reasons to Thai with explanations."""
        return [
            {
                "key": reason,
                "thai": cls.get_rejection_reason(reason),
                "explanation": cls.get_rejection_explanation(reason),
                "english": reason.replace("_", " ").title()
            }
            for reason in reasons
        ]