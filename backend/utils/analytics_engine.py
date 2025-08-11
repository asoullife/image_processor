"""
Analytics Engine
Generates detailed analytics, charts, and performance metrics
"""

from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import psutil
import GPUtil
from uuid import UUID
import logging
import numpy as np

from backend.database.models import (
    Project, ProcessingSession, ImageResult, Checkpoint
)
from backend.api.schemas import (
    AnalyticsData, ChartData, ChartDataPoint, ProcessingMetrics,
    RealtimeStats, SessionComparison
)
from backend.utils.report_generator import WebReportGenerator
from backend.utils.thai_translations import get_thai_rejection_reason

logger = logging.getLogger(__name__)

class AnalyticsEngine:
    """Generate detailed analytics and performance metrics"""
    
    def __init__(self, db: Session):
        self.db = db
        self.report_generator = WebReportGenerator(db)
    
    async def generate_analytics(self, session_id: str) -> Optional[AnalyticsData]:
        """Generate comprehensive analytics data for a session"""
        try:
            session_uuid = UUID(session_id)
            
            # Generate summary
            summary = await self.report_generator.generate_session_summary(session_id)
            if not summary:
                return None
            
            # Generate charts
            charts = await self._generate_all_charts(session_id)
            
            # Get performance metrics
            performance_metrics = await self.get_performance_metrics(session_id)
            
            # Get filter options
            filter_options = await self.report_generator.get_filter_options(session_id)
            
            # Get recent activity
            recent_activity = await self._get_recent_activity(session_id)
            
            return AnalyticsData(
                session_id=session_uuid,
                summary=summary,
                charts=charts,
                performance_metrics=performance_metrics,
                filter_options=filter_options,
                recent_activity=recent_activity
            )
            
        except Exception as e:
            logger.error(f"Error generating analytics: {str(e)}")
            return None
    
    async def generate_chart_data(self, session_id: str, chart_type: str) -> Optional[ChartData]:
        """Generate specific chart data"""
        try:
            if chart_type == "approval_rate":
                return await self._generate_approval_rate_chart(session_id)
            elif chart_type == "rejection_reasons":
                return await self._generate_rejection_reasons_chart(session_id)
            elif chart_type == "quality_distribution":
                return await self._generate_quality_distribution_chart(session_id)
            elif chart_type == "processing_timeline":
                return await self._generate_processing_timeline_chart(session_id)
            elif chart_type == "source_folder_breakdown":
                return await self._generate_source_folder_chart(session_id)
            elif chart_type == "performance_metrics":
                return await self._generate_performance_chart(session_id)
            else:
                logger.warning(f"Unknown chart type: {chart_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating chart data for {chart_type}: {str(e)}")
            return None
    
    async def get_performance_metrics(self, session_id: str) -> Optional[ProcessingMetrics]:
        """Get real-time performance metrics"""
        try:
            session_uuid = UUID(session_id)
            
            session = self.db.query(ProcessingSession).filter(
                ProcessingSession.id == session_uuid
            ).first()
            
            if not session:
                return None
            
            # Calculate processing speed
            if session.start_time and session.processed_images > 0:
                elapsed_time = (datetime.now() - session.start_time).total_seconds() / 60  # minutes
                current_speed = session.processed_images / elapsed_time if elapsed_time > 0 else 0.0
            else:
                current_speed = 0.0
            
            # Estimate completion time
            estimated_completion = None
            if current_speed > 0 and session.total_images > session.processed_images:
                remaining_images = session.total_images - session.processed_images
                remaining_minutes = remaining_images / current_speed
                estimated_completion = datetime.now() + timedelta(minutes=remaining_minutes)
            
            # Get system metrics
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Get GPU usage if available
            gpu_usage = None
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = gpus[0].load * 100
            except:
                pass
            
            # Get current batch size (from latest checkpoint or default)
            latest_checkpoint = self.db.query(Checkpoint).filter(
                Checkpoint.session_id == session_uuid
            ).order_by(Checkpoint.created_at.desc()).first()
            
            batch_size = 20  # default
            if latest_checkpoint and latest_checkpoint.session_state:
                batch_size = latest_checkpoint.session_state.get('batch_size', 20)
            
            # Count errors and warnings
            results = self.db.query(ImageResult).filter(
                ImageResult.session_id == session_uuid
            ).all()
            
            errors_count = len([r for r in results if r.final_decision == 'rejected'])
            warnings_count = len([r for r in results if r.human_override])
            
            return ProcessingMetrics(
                session_id=session_uuid,
                current_speed=current_speed,
                estimated_completion=estimated_completion,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                cpu_usage=cpu_usage,
                batch_size=batch_size,
                current_image_index=session.processed_images,
                errors_count=errors_count,
                warnings_count=warnings_count
            )
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return None
    
    async def get_realtime_stats(self, session_id: str) -> Dict[str, Any]:
        """Get real-time statistics for live dashboard updates"""
        try:
            session_uuid = UUID(session_id)
            
            session = self.db.query(ProcessingSession).filter(
                ProcessingSession.id == session_uuid
            ).first()
            
            if not session:
                return {}
            
            # Get current image being processed
            latest_result = self.db.query(ImageResult).filter(
                ImageResult.session_id == session_uuid
            ).order_by(ImageResult.created_at.desc()).first()
            
            current_image = latest_result.filename if latest_result else None
            
            # Calculate processing speed and ETA
            processing_speed = 0.0
            eta_minutes = None
            
            if session.start_time and session.processed_images > 0:
                elapsed_time = (datetime.now() - session.start_time).total_seconds() / 60
                processing_speed = session.processed_images / elapsed_time if elapsed_time > 0 else 0.0
                
                if processing_speed > 0 and session.total_images > session.processed_images:
                    remaining_images = session.total_images - session.processed_images
                    eta_minutes = remaining_images / processing_speed
            
            return {
                "session_id": str(session_uuid),
                "current_status": session.status,
                "processed_count": session.processed_images,
                "approved_count": session.approved_images,
                "rejected_count": session.rejected_images,
                "total_count": session.total_images,
                "processing_speed": processing_speed,
                "eta_minutes": eta_minutes,
                "current_image": current_image,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting real-time stats: {str(e)}")
            return {}
    
    async def compare_sessions(self, session_ids: List[str]) -> SessionComparison:
        """Compare multiple processing sessions"""
        try:
            session_uuids = [UUID(sid) for sid in session_ids]
            
            comparison_metrics = {}
            performance_data = {}
            
            for session_id in session_ids:
                summary = await self.report_generator.generate_session_summary(session_id)
                if summary:
                    comparison_metrics[session_id] = {
                        "total_images": summary.total_images,
                        "approval_rate": summary.approval_rate,
                        "processing_time": summary.processing_time,
                        "average_time_per_image": summary.average_processing_time_per_image,
                        "quality_score_average": summary.quality_score_average,
                        "human_overrides": summary.human_overrides
                    }
                    
                    performance_data[session_id] = {
                        "images_per_minute": summary.total_images / (summary.processing_time / 60) if summary.processing_time > 0 else 0,
                        "efficiency_score": summary.approval_rate * (1 - summary.human_overrides / summary.total_images) if summary.total_images > 0 else 0
                    }
            
            # Generate recommendations
            recommendations = self._generate_comparison_recommendations(comparison_metrics, performance_data)
            
            return SessionComparison(
                sessions=session_uuids,
                comparison_metrics=comparison_metrics,
                performance_comparison=performance_data,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error comparing sessions: {str(e)}")
            return SessionComparison(
                sessions=[],
                comparison_metrics={},
                performance_comparison={},
                recommendations=[]
            )
    
    async def _generate_all_charts(self, session_id: str) -> List[ChartData]:
        """Generate all chart types for the session"""
        charts = []
        
        chart_types = [
            "approval_rate",
            "rejection_reasons", 
            "quality_distribution",
            "processing_timeline",
            "source_folder_breakdown"
        ]
        
        for chart_type in chart_types:
            chart = await self.generate_chart_data(session_id, chart_type)
            if chart:
                charts.append(chart)
        
        return charts
    
    async def _generate_approval_rate_chart(self, session_id: str) -> ChartData:
        """Generate approval rate pie chart"""
        session_uuid = UUID(session_id)
        
        results = self.db.query(ImageResult).filter(
            ImageResult.session_id == session_uuid
        ).all()
        
        approved = len([r for r in results if r.final_decision == 'approved'])
        rejected = len([r for r in results if r.final_decision == 'rejected'])
        pending = len(results) - approved - rejected
        
        data_points = [
            ChartDataPoint(label="อนุมัติ", value=approved, color="#10B981", metadata={"percentage": approved/len(results)*100 if results else 0}),
            ChartDataPoint(label="ปฏิเสธ", value=rejected, color="#EF4444", metadata={"percentage": rejected/len(results)*100 if results else 0}),
            ChartDataPoint(label="รอดำเนินการ", value=pending, color="#F59E0B", metadata={"percentage": pending/len(results)*100 if results else 0})
        ]
        
        return ChartData(
            chart_type="pie",
            title="อัตราการอนุมัติภาพ",
            data=data_points,
            chart_config={
                "responsive": True,
                "plugins": {
                    "legend": {"position": "bottom"},
                    "tooltip": {"enabled": True}
                }
            }
        )
    
    async def _generate_rejection_reasons_chart(self, session_id: str) -> ChartData:
        """Generate rejection reasons bar chart"""
        session_uuid = UUID(session_id)
        
        results = self.db.query(ImageResult).filter(
            and_(
                ImageResult.session_id == session_uuid,
                ImageResult.final_decision == 'rejected'
            )
        ).all()
        
        reason_counts = {}
        for result in results:
            if result.rejection_reasons:
                for reason in result.rejection_reasons:
                    thai_reason = get_thai_rejection_reason(reason)
                    reason_counts[thai_reason] = reason_counts.get(thai_reason, 0) + 1
        
        data_points = [
            ChartDataPoint(
                label=reason,
                value=count,
                color=self._get_reason_color(reason),
                metadata={"percentage": count/len(results)*100 if results else 0}
            )
            for reason, count in reason_counts.items()
        ]
        
        return ChartData(
            chart_type="bar",
            title="สาเหตุการปฏิเสธ",
            data=data_points,
            chart_config={
                "responsive": True,
                "scales": {
                    "y": {"beginAtZero": True}
                }
            }
        )
    
    async def _generate_quality_distribution_chart(self, session_id: str) -> ChartData:
        """Generate quality score distribution histogram"""
        session_uuid = UUID(session_id)
        
        results = self.db.query(ImageResult).filter(
            ImageResult.session_id == session_uuid
        ).all()
        
        quality_scores = []
        for result in results:
            if result.quality_scores and 'overall_score' in result.quality_scores:
                quality_scores.append(result.quality_scores['overall_score'])
        
        if not quality_scores:
            return ChartData(
                chart_type="histogram",
                title="การกระจายคะแนนคุณภาพ",
                data=[],
                chart_config={}
            )
        
        # Create histogram bins
        bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
        hist, bin_edges = np.histogram(quality_scores, bins=bins)
        
        data_points = []
        for i in range(len(hist)):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            label = f"{bin_start:.1f}-{bin_end:.1f}"
            data_points.append(
                ChartDataPoint(
                    label=label,
                    value=int(hist[i]),
                    color="#3B82F6",
                    metadata={"bin_start": bin_start, "bin_end": bin_end}
                )
            )
        
        return ChartData(
            chart_type="histogram",
            title="การกระจายคะแนนคุณภาพ",
            data=data_points,
            chart_config={
                "responsive": True,
                "scales": {
                    "x": {"title": {"display": True, "text": "คะแนนคุณภาพ"}},
                    "y": {"title": {"display": True, "text": "จำนวนภาพ"}}
                }
            }
        )
    
    async def _generate_processing_timeline_chart(self, session_id: str) -> ChartData:
        """Generate processing timeline chart"""
        session_uuid = UUID(session_id)
        
        results = self.db.query(ImageResult).filter(
            ImageResult.session_id == session_uuid
        ).order_by(ImageResult.created_at).all()
        
        if not results:
            return ChartData(
                chart_type="line",
                title="ไทม์ไลน์การประมวลผล",
                data=[],
                chart_config={}
            )
        
        # Group by hour
        hourly_counts = {}
        for result in results:
            hour_key = result.created_at.replace(minute=0, second=0, microsecond=0)
            if hour_key not in hourly_counts:
                hourly_counts[hour_key] = {"approved": 0, "rejected": 0}
            
            if result.final_decision == 'approved':
                hourly_counts[hour_key]["approved"] += 1
            elif result.final_decision == 'rejected':
                hourly_counts[hour_key]["rejected"] += 1
        
        # Create data points
        data_points = []
        for hour, counts in sorted(hourly_counts.items()):
            data_points.append(
                ChartDataPoint(
                    label=hour.strftime("%H:%M"),
                    value=counts["approved"] + counts["rejected"],
                    metadata={
                        "approved": counts["approved"],
                        "rejected": counts["rejected"],
                        "timestamp": hour.isoformat()
                    }
                )
            )
        
        return ChartData(
            chart_type="line",
            title="ไทม์ไลน์การประมวลผล",
            data=data_points,
            chart_config={
                "responsive": True,
                "scales": {
                    "x": {"title": {"display": True, "text": "เวลา"}},
                    "y": {"title": {"display": True, "text": "จำนวนภาพที่ประมวลผล"}}
                }
            }
        )
    
    async def _generate_source_folder_chart(self, session_id: str) -> ChartData:
        """Generate source folder breakdown chart"""
        session_uuid = UUID(session_id)
        
        results = self.db.query(ImageResult).filter(
            ImageResult.session_id == session_uuid
        ).all()
        
        folder_stats = {}
        for result in results:
            folder = result.source_folder or "Unknown"
            if folder not in folder_stats:
                folder_stats[folder] = {"total": 0, "approved": 0, "rejected": 0}
            
            folder_stats[folder]["total"] += 1
            if result.final_decision == 'approved':
                folder_stats[folder]["approved"] += 1
            elif result.final_decision == 'rejected':
                folder_stats[folder]["rejected"] += 1
        
        data_points = []
        for folder, stats in folder_stats.items():
            approval_rate = stats["approved"] / stats["total"] if stats["total"] > 0 else 0
            data_points.append(
                ChartDataPoint(
                    label=f"โฟลเดอร์ {folder}",
                    value=approval_rate * 100,
                    color=self._get_folder_color(folder),
                    metadata={
                        "total": stats["total"],
                        "approved": stats["approved"],
                        "rejected": stats["rejected"]
                    }
                )
            )
        
        return ChartData(
            chart_type="bar",
            title="อัตราการอนุมัติตามโฟลเดอร์",
            data=data_points,
            chart_config={
                "responsive": True,
                "scales": {
                    "y": {"beginAtZero": True, "max": 100, "title": {"display": True, "text": "อัตราการอนุมัติ (%)"}}
                }
            }
        )
    
    async def _get_recent_activity(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent activity for the session"""
        try:
            session_uuid = UUID(session_id)
            
            recent_results = self.db.query(ImageResult).filter(
                ImageResult.session_id == session_uuid
            ).order_by(ImageResult.created_at.desc()).limit(limit).all()
            
            activity = []
            for result in recent_results:
                activity_type = "processed"
                if result.human_override:
                    activity_type = "human_override"
                elif result.final_decision == 'approved':
                    activity_type = "approved"
                elif result.final_decision == 'rejected':
                    activity_type = "rejected"
                
                activity.append({
                    "type": activity_type,
                    "filename": result.filename,
                    "decision": result.final_decision,
                    "timestamp": result.created_at.isoformat(),
                    "processing_time": result.processing_time
                })
            
            return activity
            
        except Exception as e:
            logger.error(f"Error getting recent activity: {str(e)}")
            return []
    
    def _get_reason_color(self, reason: str) -> str:
        """Get color for rejection reason"""
        color_map = {
            "คุณภาพภาพต่ำ ไม่เหมาะสำหรับขาย": "#EF4444",
            "พบความผิดปกติในภาพ": "#F59E0B",
            "พบภาพที่คล้ายกัน อาจถูกปฏิเสธเป็น spam": "#8B5CF6",
            "ปัญหาลิขสิทธิ์หรือความเป็นส่วนตัว": "#EC4899",
            "ปัญหาทางเทคนิค": "#6B7280"
        }
        return color_map.get(reason, "#9CA3AF")
    
    def _get_folder_color(self, folder: str) -> str:
        """Get color for source folder"""
        colors = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]
        folder_hash = hash(folder) % len(colors)
        return colors[folder_hash]
    
    def _generate_comparison_recommendations(
        self, 
        comparison_metrics: Dict[str, Dict[str, float]], 
        performance_data: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """Generate recommendations based on session comparison"""
        recommendations = []
        
        if not comparison_metrics:
            return recommendations
        
        # Find best performing session
        best_approval_rate = max(
            (metrics.get("approval_rate", 0) for metrics in comparison_metrics.values()),
            default=0
        )
        
        best_efficiency = max(
            (perf.get("efficiency_score", 0) for perf in performance_data.values()),
            default=0
        )
        
        # Generate recommendations
        if best_approval_rate > 0.8:
            recommendations.append("เซสชันที่มีอัตราการอนุมัติสูงสุดแสดงให้เห็นถึงการตั้งค่าที่เหมาะสม")
        
        if best_efficiency > 0.7:
            recommendations.append("ควรใช้การตั้งค่าจากเซสชันที่มีประสิทธิภาพสูงสุดเป็นแม่แบบ")
        
        # Check for patterns
        avg_approval_rate = np.mean([metrics.get("approval_rate", 0) for metrics in comparison_metrics.values()])
        if avg_approval_rate < 0.5:
            recommendations.append("อัตราการอนุมัติโดยรวมต่ำ ควรปรับการตั้งค่าคุณภาพหรือตรวจสอบข้อมูลนำเข้า")
        
        return recommendations