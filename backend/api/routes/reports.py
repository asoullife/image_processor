"""
Reports and Analytics API Routes
Provides comprehensive web-based reporting with real-time updates
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json

from backend.database.connection import get_db
from backend.api.schemas import (
    ReportSummary, 
    AnalyticsData, 
    ChartData, 
    FilterOptions,
    ProcessingMetrics,
    ThumbnailData
)
from backend.utils.report_generator import WebReportGenerator
from backend.utils.analytics_engine import AnalyticsEngine
from backend.utils.thumbnail_generator import ThumbnailGenerator

router = APIRouter(prefix="/api/reports", tags=["reports"])

@router.get("/summary/{session_id}", response_model=ReportSummary)
async def get_session_summary(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Get comprehensive summary statistics for a processing session"""
    try:
        report_generator = WebReportGenerator(db)
        summary = await report_generator.generate_session_summary(session_id)
        
        if not summary:
            raise HTTPException(status_code=404, detail="Session not found")
            
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

@router.get("/analytics/{session_id}", response_model=AnalyticsData)
async def get_session_analytics(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Get detailed analytics data including charts and performance metrics"""
    try:
        analytics_engine = AnalyticsEngine(db)
        analytics = await analytics_engine.generate_analytics(session_id)
        
        if not analytics:
            raise HTTPException(status_code=404, detail="Session not found")
            
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating analytics: {str(e)}")

@router.get("/charts/{session_id}")
async def get_chart_data(
    session_id: str,
    chart_type: str = Query(..., description="Chart type: approval_rate, rejection_reasons, quality_distribution, processing_timeline"),
    db: Session = Depends(get_db)
):
    """Get specific chart data for visualization"""
    try:
        analytics_engine = AnalyticsEngine(db)
        chart_data = await analytics_engine.generate_chart_data(session_id, chart_type)
        
        if not chart_data:
            raise HTTPException(status_code=404, detail="Chart data not found")
            
        return chart_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating chart data: {str(e)}")

@router.get("/filters/{session_id}", response_model=FilterOptions)
async def get_filter_options(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Get available filter options for the session"""
    try:
        report_generator = WebReportGenerator(db)
        filters = await report_generator.get_filter_options(session_id)
        
        return filters
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting filter options: {str(e)}")

@router.get("/results/{session_id}")
async def get_filtered_results(
    session_id: str,
    decision: Optional[str] = Query(None, description="Filter by decision: approved, rejected, pending"),
    rejection_reason: Optional[str] = Query(None, description="Filter by rejection reason"),
    source_folder: Optional[str] = Query(None, description="Filter by source folder"),
    human_override: Optional[bool] = Query(None, description="Filter by human override status"),
    search: Optional[str] = Query(None, description="Search in filenames"),
    sort_by: str = Query("created_at", description="Sort by: filename, created_at, confidence, processing_time"),
    sort_order: str = Query("desc", description="Sort order: asc, desc"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page"),
    db: Session = Depends(get_db)
):
    """Get filtered and paginated processing results"""
    try:
        report_generator = WebReportGenerator(db)
        results = await report_generator.get_filtered_results(
            session_id=session_id,
            filters={
                "decision": decision,
                "rejection_reason": rejection_reason,
                "source_folder": source_folder,
                "human_override": human_override,
                "search": search
            },
            sort_by=sort_by,
            sort_order=sort_order,
            page=page,
            page_size=page_size
        )
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting filtered results: {str(e)}")

@router.get("/performance/{session_id}", response_model=ProcessingMetrics)
async def get_performance_metrics(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Get real-time performance metrics for the session"""
    try:
        analytics_engine = AnalyticsEngine(db)
        metrics = await analytics_engine.get_performance_metrics(session_id)
        
        if not metrics:
            raise HTTPException(status_code=404, detail="Session not found")
            
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting performance metrics: {str(e)}")

@router.get("/thumbnails/{session_id}")
async def get_thumbnail_data(
    session_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """Get thumbnail data for image grid display"""
    try:
        thumbnail_generator = ThumbnailGenerator(db)
        thumbnails = await thumbnail_generator.get_session_thumbnails(
            session_id=session_id,
            page=page,
            page_size=page_size
        )
        
        return thumbnails
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting thumbnails: {str(e)}")

@router.get("/export/{session_id}")
async def export_report(
    session_id: str,
    format: str = Query("json", description="Export format: json, csv, excel"),
    db: Session = Depends(get_db)
):
    """Export complete report in specified format"""
    try:
        report_generator = WebReportGenerator(db)
        
        if format == "json":
            export_data = await report_generator.export_json(session_id)
            return {"data": export_data, "filename": f"report_{session_id}.json"}
        elif format == "csv":
            csv_data = await report_generator.export_csv(session_id)
            return {"data": csv_data, "filename": f"report_{session_id}.csv"}
        elif format == "excel":
            excel_data = await report_generator.export_excel(session_id)
            return {"data": excel_data, "filename": f"report_{session_id}.xlsx"}
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting report: {str(e)}")

@router.get("/realtime/{session_id}")
async def get_realtime_stats(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Get real-time statistics for live dashboard updates"""
    try:
        analytics_engine = AnalyticsEngine(db)
        realtime_stats = await analytics_engine.get_realtime_stats(session_id)
        
        return realtime_stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting real-time stats: {str(e)}")

@router.get("/comparison")
async def compare_sessions(
    session_ids: List[str] = Query(..., description="List of session IDs to compare"),
    db: Session = Depends(get_db)
):
    """Compare multiple processing sessions"""
    try:
        analytics_engine = AnalyticsEngine(db)
        comparison = await analytics_engine.compare_sessions(session_ids)
        
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing sessions: {str(e)}")