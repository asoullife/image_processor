"""
Mock Reports API for Testing
Provides mock data for testing the web-based reports system
"""

from fastapi import APIRouter
from typing import Dict, Any, List
from datetime import datetime, timedelta
import random
from uuid import uuid4

router = APIRouter(prefix="/api/reports", tags=["mock-reports"])

# Mock data generators
def generate_mock_summary(session_id: str) -> Dict[str, Any]:
    """Generate mock summary data"""
    total_images = random.randint(1000, 5000)
    processed_images = random.randint(800, total_images)
    approved_images = random.randint(int(processed_images * 0.6), int(processed_images * 0.9))
    rejected_images = processed_images - approved_images
    
    return {
        "session_id": session_id,
        "total_images": total_images,
        "processed_images": processed_images,
        "approved_images": approved_images,
        "rejected_images": rejected_images,
        "pending_images": total_images - processed_images,
        "approval_rate": approved_images / processed_images if processed_images > 0 else 0,
        "processing_time": random.randint(1800, 7200),  # 30 minutes to 2 hours
        "average_processing_time_per_image": random.uniform(1.5, 4.0),
        "rejection_breakdown": {
            "คุณภาพภาพต่ำ ไม่เหมาะสำหรับขาย": random.randint(50, 200),
            "พบความผิดปกติในภาพ": random.randint(20, 100),
            "พบภาพที่คล้ายกัน อาจถูกปฏิเสธเป็น spam": random.randint(10, 80),
            "ปัญหาลิขสิทธิ์หรือความเป็นส่วนตัว": random.randint(5, 50),
            "ปัญหาทางเทคนิค": random.randint(1, 20)
        },
        "quality_score_average": random.uniform(0.65, 0.85),
        "human_overrides": random.randint(10, 100),
        "created_at": (datetime.now() - timedelta(hours=2)).isoformat(),
        "updated_at": datetime.now().isoformat()
    }

def generate_mock_charts() -> List[Dict[str, Any]]:
    """Generate mock chart data"""
    return [
        {
            "chart_type": "pie",
            "title": "อัตราการอนุมัติภาพ",
            "data": [
                {"label": "อนุมัติ", "value": 1250, "color": "#10B981", "metadata": {"percentage": 75.0}},
                {"label": "ปฏิเสธ", "value": 350, "color": "#EF4444", "metadata": {"percentage": 21.0}},
                {"label": "รอดำเนินการ", "value": 67, "color": "#F59E0B", "metadata": {"percentage": 4.0}}
            ],
            "chart_config": {
                "responsive": True,
                "plugins": {"legend": {"position": "bottom"}}
            }
        },
        {
            "chart_type": "bar",
            "title": "สาเหตุการปฏิเสธ",
            "data": [
                {"label": "คุณภาพต่ำ", "value": 150, "color": "#EF4444"},
                {"label": "ความผิดปกติ", "value": 80, "color": "#F59E0B"},
                {"label": "ภาพซ้ำ", "value": 60, "color": "#8B5CF6"},
                {"label": "ลิขสิทธิ์", "value": 40, "color": "#EC4899"},
                {"label": "เทคนิค", "value": 20, "color": "#6B7280"}
            ],
            "chart_config": {
                "responsive": True,
                "scales": {"y": {"beginAtZero": True}}
            }
        },
        {
            "chart_type": "histogram",
            "title": "การกระจายคะแนนคุณภาพ",
            "data": [
                {"label": "0.0-0.1", "value": 5, "color": "#3B82F6"},
                {"label": "0.1-0.2", "value": 12, "color": "#3B82F6"},
                {"label": "0.2-0.3", "value": 25, "color": "#3B82F6"},
                {"label": "0.3-0.4", "value": 45, "color": "#3B82F6"},
                {"label": "0.4-0.5", "value": 80, "color": "#3B82F6"},
                {"label": "0.5-0.6", "value": 120, "color": "#3B82F6"},
                {"label": "0.6-0.7", "value": 200, "color": "#3B82F6"},
                {"label": "0.7-0.8", "value": 350, "color": "#3B82F6"},
                {"label": "0.8-0.9", "value": 280, "color": "#3B82F6"},
                {"label": "0.9-1.0", "value": 150, "color": "#3B82F6"}
            ],
            "chart_config": {
                "responsive": True,
                "scales": {
                    "x": {"title": {"display": True, "text": "คะแนนคุณภาพ"}},
                    "y": {"title": {"display": True, "text": "จำนวนภาพ"}}
                }
            }
        }
    ]

def generate_mock_performance_metrics(session_id: str) -> Dict[str, Any]:
    """Generate mock performance metrics"""
    return {
        "session_id": session_id,
        "current_speed": random.uniform(15.0, 45.0),
        "estimated_completion": (datetime.now() + timedelta(minutes=random.randint(30, 180))).isoformat(),
        "memory_usage": random.uniform(45.0, 85.0),
        "gpu_usage": random.uniform(60.0, 95.0),
        "cpu_usage": random.uniform(30.0, 80.0),
        "batch_size": random.choice([10, 20, 30, 50]),
        "current_image_index": random.randint(800, 1500),
        "errors_count": random.randint(0, 15),
        "warnings_count": random.randint(5, 50)
    }

def generate_mock_filter_options() -> Dict[str, Any]:
    """Generate mock filter options"""
    return {
        "decisions": ["approved", "rejected", "pending"],
        "rejection_reasons": [
            "คุณภาพภาพต่ำ ไม่เหมาะสำหรับขาย",
            "พบความผิดปกติในภาพ",
            "พบภาพที่คล้ายกัน อาจถูกปฏิเสธเป็น spam",
            "ปัญหาลิขสิทธิ์หรือความเป็นส่วนตัว",
            "ปัญหาทางเทคนิค"
        ],
        "source_folders": ["1", "2", "3", "4", "5"],
        "date_range": {
            "min": (datetime.now() - timedelta(days=7)).isoformat(),
            "max": datetime.now().isoformat()
        },
        "quality_score_range": {
            "min": 0.0,
            "max": 1.0
        }
    }

def generate_mock_recent_activity() -> List[Dict[str, Any]]:
    """Generate mock recent activity"""
    activities = []
    activity_types = ["approved", "rejected", "human_override", "processed"]
    
    for i in range(10):
        activities.append({
            "type": random.choice(activity_types),
            "filename": f"IMG_{random.randint(1000, 9999)}.jpg",
            "decision": random.choice(["approved", "rejected"]),
            "timestamp": (datetime.now() - timedelta(minutes=random.randint(1, 120))).isoformat(),
            "processing_time": random.uniform(1.0, 5.0)
        })
    
    return activities

# Mock API endpoints
@router.get("/analytics/{session_id}")
async def get_mock_analytics(session_id: str):
    """Get mock analytics data"""
    summary = generate_mock_summary(session_id)
    charts = generate_mock_charts()
    performance_metrics = generate_mock_performance_metrics(session_id)
    filter_options = generate_mock_filter_options()
    recent_activity = generate_mock_recent_activity()
    
    return {
        "session_id": session_id,
        "summary": summary,
        "charts": charts,
        "performance_metrics": performance_metrics,
        "filter_options": filter_options,
        "recent_activity": recent_activity
    }

@router.get("/realtime/{session_id}")
async def get_mock_realtime_stats(session_id: str):
    """Get mock real-time statistics"""
    return {
        "session_id": session_id,
        "current_status": "running",
        "processed_count": random.randint(800, 1500),
        "approved_count": random.randint(600, 1200),
        "rejected_count": random.randint(100, 300),
        "processing_speed": random.uniform(15.0, 45.0),
        "eta_minutes": random.randint(30, 180),
        "current_image": f"IMG_{random.randint(1000, 9999)}.jpg",
        "last_updated": datetime.now().isoformat()
    }

@router.get("/results/{session_id}")
async def get_mock_filtered_results(
    session_id: str,
    page: int = 1,
    page_size: int = 50
):
    """Get mock filtered results"""
    # Generate mock results
    total_count = random.randint(1000, 2000)
    items = []
    
    for i in range(min(page_size, total_count)):
        item_id = str(uuid4())
        decision = random.choice(["approved", "rejected", "pending"])
        
        rejection_reasons = []
        if decision == "rejected":
            rejection_reasons = random.sample([
                "คุณภาพภาพต่ำ ไม่เหมาะสำหรับขาย",
                "พบความผิดปกติในภาพ",
                "พบภาพที่คล้ายกัน อาจถูกปฏิเสธเป็น spam"
            ], random.randint(1, 2))
        
        items.append({
            "id": item_id,
            "filename": f"IMG_{random.randint(1000, 9999)}.jpg",
            "image_path": f"/mock/images/IMG_{random.randint(1000, 9999)}.jpg",
            "source_folder": str(random.randint(1, 5)),
            "final_decision": decision,
            "rejection_reasons": rejection_reasons,
            "quality_scores": {
                "overall_score": random.uniform(0.3, 0.95)
            },
            "human_override": random.choice([True, False]),
            "processing_time": random.uniform(1.0, 5.0),
            "created_at": (datetime.now() - timedelta(minutes=random.randint(1, 1440))).isoformat(),
            "thumbnail_url": f"/api/thumbnails/{item_id}"
        })
    
    return {
        "items": items,
        "total_count": total_count,
        "page": page,
        "page_size": page_size,
        "total_pages": (total_count + page_size - 1) // page_size
    }

@router.get("/thumbnails/{session_id}")
async def get_mock_thumbnails(
    session_id: str,
    page: int = 1,
    page_size: int = 100
):
    """Get mock thumbnail data"""
    total_count = random.randint(500, 1500)
    items = []
    
    for i in range(min(page_size, total_count)):
        item_id = str(uuid4())
        decision = random.choice(["approved", "rejected", "pending"])
        
        rejection_reasons = []
        if decision == "rejected":
            rejection_reasons = random.sample([
                "คุณภาพภาพต่ำ ไม่เหมาะสำหรับขาย",
                "พบความผิดปกติในภาพ"
            ], random.randint(1, 2))
        
        items.append({
            "image_id": item_id,
            "filename": f"IMG_{random.randint(1000, 9999)}.jpg",
            "thumbnail_url": f"https://picsum.photos/200/200?random={i}",  # Use Lorem Picsum for demo
            "decision": decision,
            "rejection_reasons": rejection_reasons,
            "confidence_scores": {
                "overall_score": random.uniform(0.3, 0.95)
            },
            "human_override": random.choice([True, False]),
            "processing_time": random.uniform(1.0, 5.0),
            "source_folder": str(random.randint(1, 5)),
            "created_at": (datetime.now() - timedelta(minutes=random.randint(1, 1440))).isoformat()
        })
    
    return {
        "items": items,
        "total_count": total_count,
        "page": page,
        "page_size": page_size,
        "total_pages": (total_count + page_size - 1) // page_size
    }

@router.get("/export/{session_id}")
async def mock_export_report(session_id: str, format: str = "json"):
    """Mock export functionality"""
    if format == "json":
        return {
            "data": {"mock": "json_data", "session_id": session_id},
            "filename": f"report_{session_id}.json"
        }
    elif format == "csv":
        return {
            "data": "filename,decision,quality_score\nIMG_001.jpg,approved,0.85\nIMG_002.jpg,rejected,0.45",
            "filename": f"report_{session_id}.csv"
        }
    elif format == "excel":
        return {
            "data": "mock_excel_base64_data",
            "filename": f"report_{session_id}.xlsx"
        }
    else:
        return {"error": "Unsupported format"}