"""
Web Report Generator
Generates comprehensive web-based reports and analytics
"""

from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import pandas as pd
from uuid import UUID
import logging

from backend.database.models import (
    Project, ProcessingSession, ImageResult, Checkpoint
)
from backend.api.schemas import (
    ReportSummary, FilterOptions, ThumbnailData
)
from backend.utils.thai_translations import get_thai_rejection_reason

logger = logging.getLogger(__name__)

class WebReportGenerator:
    """Generate comprehensive web-based reports and analytics"""
    
    def __init__(self, db: Session):
        self.db = db
    
    async def generate_session_summary(self, session_id: str) -> Optional[ReportSummary]:
        """Generate comprehensive summary statistics for a session"""
        try:
            session_uuid = UUID(session_id)
            
            # Get session data
            session = self.db.query(ProcessingSession).filter(
                ProcessingSession.id == session_uuid
            ).first()
            
            if not session:
                return None
            
            # Get image results
            results = self.db.query(ImageResult).filter(
                ImageResult.session_id == session_uuid
            ).all()
            
            if not results:
                return ReportSummary(
                    session_id=session_uuid,
                    total_images=0,
                    processed_images=0,
                    approved_images=0,
                    rejected_images=0,
                    pending_images=0,
                    approval_rate=0.0,
                    processing_time=0.0,
                    average_processing_time_per_image=0.0,
                    rejection_breakdown={},
                    quality_score_average=0.0,
                    human_overrides=0,
                    created_at=session.created_at,
                    updated_at=session.updated_at
                )
            
            # Calculate statistics
            total_images = len(results)
            approved_images = len([r for r in results if r.final_decision == 'approved'])
            rejected_images = len([r for r in results if r.final_decision == 'rejected'])
            pending_images = total_images - approved_images - rejected_images
            
            approval_rate = approved_images / total_images if total_images > 0 else 0.0
            
            # Calculate processing times
            processing_times = [r.processing_time for r in results if r.processing_time]
            total_processing_time = sum(processing_times) if processing_times else 0.0
            avg_processing_time = total_processing_time / len(processing_times) if processing_times else 0.0
            
            # Rejection reason breakdown
            rejection_breakdown = {}
            for result in results:
                if result.final_decision == 'rejected' and result.rejection_reasons:
                    for reason in result.rejection_reasons:
                        thai_reason = get_thai_rejection_reason(reason)
                        rejection_breakdown[thai_reason] = rejection_breakdown.get(thai_reason, 0) + 1
            
            # Quality score average
            quality_scores = []
            for result in results:
                if result.quality_scores and 'overall_score' in result.quality_scores:
                    quality_scores.append(result.quality_scores['overall_score'])
            
            quality_score_average = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            
            # Human overrides count
            human_overrides = len([r for r in results if r.human_override])
            
            return ReportSummary(
                session_id=session_uuid,
                total_images=total_images,
                processed_images=session.processed_images,
                approved_images=approved_images,
                rejected_images=rejected_images,
                pending_images=pending_images,
                approval_rate=approval_rate,
                processing_time=total_processing_time,
                average_processing_time_per_image=avg_processing_time,
                rejection_breakdown=rejection_breakdown,
                quality_score_average=quality_score_average,
                human_overrides=human_overrides,
                created_at=session.created_at,
                updated_at=session.updated_at
            )
            
        except Exception as e:
            logger.error(f"Error generating session summary: {str(e)}")
            return None
    
    async def get_filter_options(self, session_id: str) -> FilterOptions:
        """Get available filter options for the session"""
        try:
            session_uuid = UUID(session_id)
            
            # Get unique values from results
            results = self.db.query(ImageResult).filter(
                ImageResult.session_id == session_uuid
            ).all()
            
            decisions = list(set([r.final_decision for r in results if r.final_decision]))
            
            rejection_reasons = set()
            for result in results:
                if result.rejection_reasons:
                    for reason in result.rejection_reasons:
                        thai_reason = get_thai_rejection_reason(reason)
                        rejection_reasons.add(thai_reason)
            
            source_folders = list(set([r.source_folder for r in results if r.source_folder]))
            
            # Date range
            dates = [r.created_at for r in results]
            date_range = {
                "min": min(dates) if dates else datetime.now(),
                "max": max(dates) if dates else datetime.now()
            }
            
            # Quality score range
            quality_scores = []
            for result in results:
                if result.quality_scores and 'overall_score' in result.quality_scores:
                    quality_scores.append(result.quality_scores['overall_score'])
            
            quality_score_range = {
                "min": min(quality_scores) if quality_scores else 0.0,
                "max": max(quality_scores) if quality_scores else 1.0
            }
            
            return FilterOptions(
                decisions=decisions,
                rejection_reasons=list(rejection_reasons),
                source_folders=source_folders,
                date_range=date_range,
                quality_score_range=quality_score_range
            )
            
        except Exception as e:
            logger.error(f"Error getting filter options: {str(e)}")
            return FilterOptions(
                decisions=[],
                rejection_reasons=[],
                source_folders=[],
                date_range={"min": datetime.now(), "max": datetime.now()},
                quality_score_range={"min": 0.0, "max": 1.0}
            )
    
    async def get_filtered_results(
        self,
        session_id: str,
        filters: Dict[str, Any],
        sort_by: str = "created_at",
        sort_order: str = "desc",
        page: int = 1,
        page_size: int = 50
    ) -> Dict[str, Any]:
        """Get filtered and paginated processing results"""
        try:
            session_uuid = UUID(session_id)
            
            # Build query
            query = self.db.query(ImageResult).filter(
                ImageResult.session_id == session_uuid
            )
            
            # Apply filters
            if filters.get("decision"):
                query = query.filter(ImageResult.final_decision == filters["decision"])
            
            if filters.get("rejection_reason"):
                # Convert Thai reason back to English for database query
                english_reason = self._get_english_rejection_reason(filters["rejection_reason"])
                query = query.filter(ImageResult.rejection_reasons.contains([english_reason]))
            
            if filters.get("source_folder"):
                query = query.filter(ImageResult.source_folder == filters["source_folder"])
            
            if filters.get("human_override") is not None:
                query = query.filter(ImageResult.human_override == filters["human_override"])
            
            if filters.get("search"):
                search_term = f"%{filters['search']}%"
                query = query.filter(ImageResult.filename.ilike(search_term))
            
            # Apply sorting
            if sort_by == "filename":
                order_column = ImageResult.filename
            elif sort_by == "confidence":
                # Sort by overall quality score
                order_column = func.cast(
                    func.json_extract(ImageResult.quality_scores, '$.overall_score'),
                    func.Float
                )
            elif sort_by == "processing_time":
                order_column = ImageResult.processing_time
            else:  # created_at
                order_column = ImageResult.created_at
            
            if sort_order == "asc":
                query = query.order_by(order_column.asc())
            else:
                query = query.order_by(order_column.desc())
            
            # Get total count
            total_count = query.count()
            
            # Apply pagination
            offset = (page - 1) * page_size
            results = query.offset(offset).limit(page_size).all()
            
            # Convert to response format
            items = []
            for result in results:
                # Convert rejection reasons to Thai
                thai_rejection_reasons = []
                if result.rejection_reasons:
                    thai_rejection_reasons = [
                        get_thai_rejection_reason(reason) for reason in result.rejection_reasons
                    ]
                
                items.append({
                    "id": str(result.id),
                    "filename": result.filename,
                    "image_path": result.image_path,
                    "source_folder": result.source_folder,
                    "final_decision": result.final_decision,
                    "rejection_reasons": thai_rejection_reasons,
                    "quality_scores": result.quality_scores,
                    "human_override": result.human_override,
                    "processing_time": result.processing_time,
                    "created_at": result.created_at,
                    "thumbnail_url": f"/api/thumbnails/{result.id}"
                })
            
            return {
                "items": items,
                "total_count": total_count,
                "page": page,
                "page_size": page_size,
                "total_pages": (total_count + page_size - 1) // page_size
            }
            
        except Exception as e:
            logger.error(f"Error getting filtered results: {str(e)}")
            return {
                "items": [],
                "total_count": 0,
                "page": page,
                "page_size": page_size,
                "total_pages": 0
            }
    
    async def export_json(self, session_id: str) -> Dict[str, Any]:
        """Export complete report as JSON"""
        try:
            session_uuid = UUID(session_id)
            
            # Get session and results
            session = self.db.query(ProcessingSession).filter(
                ProcessingSession.id == session_uuid
            ).first()
            
            if not session:
                return {}
            
            results = self.db.query(ImageResult).filter(
                ImageResult.session_id == session_uuid
            ).all()
            
            # Generate summary
            summary = await self.generate_session_summary(session_id)
            
            # Convert results to exportable format
            export_results = []
            for result in results:
                thai_rejection_reasons = []
                if result.rejection_reasons:
                    thai_rejection_reasons = [
                        get_thai_rejection_reason(reason) for reason in result.rejection_reasons
                    ]
                
                export_results.append({
                    "filename": result.filename,
                    "image_path": result.image_path,
                    "source_folder": result.source_folder,
                    "final_decision": result.final_decision,
                    "rejection_reasons": thai_rejection_reasons,
                    "quality_scores": result.quality_scores,
                    "defect_results": result.defect_results,
                    "compliance_results": result.compliance_results,
                    "human_override": result.human_override,
                    "processing_time": result.processing_time,
                    "created_at": result.created_at.isoformat()
                })
            
            return {
                "session_id": session_id,
                "summary": summary.dict() if summary else {},
                "results": export_results,
                "exported_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error exporting JSON: {str(e)}")
            return {}
    
    async def export_csv(self, session_id: str) -> str:
        """Export results as CSV"""
        try:
            session_uuid = UUID(session_id)
            
            results = self.db.query(ImageResult).filter(
                ImageResult.session_id == session_uuid
            ).all()
            
            # Prepare data for CSV
            csv_data = []
            for result in results:
                thai_rejection_reasons = []
                if result.rejection_reasons:
                    thai_rejection_reasons = [
                        get_thai_rejection_reason(reason) for reason in result.rejection_reasons
                    ]
                
                quality_score = 0.0
                if result.quality_scores and 'overall_score' in result.quality_scores:
                    quality_score = result.quality_scores['overall_score']
                
                csv_data.append({
                    "filename": result.filename,
                    "source_folder": result.source_folder,
                    "final_decision": result.final_decision,
                    "rejection_reasons": "; ".join(thai_rejection_reasons),
                    "quality_score": quality_score,
                    "human_override": result.human_override,
                    "processing_time": result.processing_time,
                    "created_at": result.created_at.isoformat()
                })
            
            # Convert to CSV
            df = pd.DataFrame(csv_data)
            return df.to_csv(index=False)
            
        except Exception as e:
            logger.error(f"Error exporting CSV: {str(e)}")
            return ""
    
    async def export_excel(self, session_id: str) -> bytes:
        """Export results as Excel"""
        try:
            session_uuid = UUID(session_id)
            
            results = self.db.query(ImageResult).filter(
                ImageResult.session_id == session_uuid
            ).all()
            
            # Prepare data for Excel
            excel_data = []
            for result in results:
                thai_rejection_reasons = []
                if result.rejection_reasons:
                    thai_rejection_reasons = [
                        get_thai_rejection_reason(reason) for reason in result.rejection_reasons
                    ]
                
                quality_score = 0.0
                if result.quality_scores and 'overall_score' in result.quality_scores:
                    quality_score = result.quality_scores['overall_score']
                
                excel_data.append({
                    "ชื่อไฟล์": result.filename,
                    "โฟลเดอร์ต้นทาง": result.source_folder,
                    "ผลการตัดสิน": result.final_decision,
                    "เหตุผลที่ปฏิเสธ": "; ".join(thai_rejection_reasons),
                    "คะแนนคุณภาพ": quality_score,
                    "แก้ไขโดยมนุษย์": result.human_override,
                    "เวลาประมวลผล (วินาที)": result.processing_time,
                    "วันที่สร้าง": result.created_at
                })
            
            # Convert to Excel
            df = pd.DataFrame(excel_data)
            
            # Create Excel file in memory
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='ผลการประมวลผล', index=False)
            
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Error exporting Excel: {str(e)}")
            return b""
    
    def _get_english_rejection_reason(self, thai_reason: str) -> str:
        """Convert Thai rejection reason back to English for database queries"""
        reason_mapping = {
            "คุณภาพภาพต่ำ ไม่เหมาะสำหรับขาย": "low_quality",
            "พบความผิดปกติในภาพ": "defect_detected",
            "พบภาพที่คล้ายกัน อาจถูกปฏิเสธเป็น spam": "similar_image",
            "ปัญหาลิขสิทธิ์หรือความเป็นส่วนตัว": "compliance_issue",
            "ปัญหาทางเทคนิค": "technical_issue"
        }
        return reason_mapping.get(thai_reason, thai_reason)