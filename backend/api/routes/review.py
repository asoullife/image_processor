"""Human review API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID

from backend.database.connection import get_db
from backend.api.schemas import (
    ImageResultResponse,
    ReviewDecision,
    BulkReviewRequest
)
from backend.database.models import ImageResult, ProcessingSession
from backend.core.output_manager import OutputManager
from backend.utils.thai_translations import ThaiTranslations

router = APIRouter(prefix="/api/review", tags=["review"])

# Thai language rejection reason translations
THAI_REJECTION_REASONS = {
    "low_quality": "คุณภาพภาพต่ำ ไม่เหมาะสำหรับขาย",
    "defects_detected": "พบความผิดปกติในภาพ",
    "similar_images": "พบภาพที่คล้ายกัน อาจถูกปฏิเสธเป็น spam",
    "compliance_issues": "ปัญหาลิขสิทธิ์หรือความเป็นส่วนตัว",
    "technical_issues": "ปัญหาทางเทคนิค",
    "metadata_issues": "ปัญหาข้อมูลเมตาดาต้า",
    "inappropriate_content": "เนื้อหาไม่เหมาะสม",
    "copyright_violation": "ละเมิดลิขสิทธิ์",
    "privacy_violation": "ละเมิดความเป็นส่วนตัว",
    "trademark_detected": "พบเครื่องหมายการค้า"
}

@router.get("/sessions/{session_id}/results")
async def get_review_results(
    session_id: UUID,
    db: Session = Depends(get_db),
    decision: Optional[str] = Query(None, description="Filter by decision: approved, rejected, pending"),
    rejection_reason: Optional[str] = Query(None, description="Filter by rejection reason"),
    human_reviewed: Optional[bool] = Query(None, description="Filter by human review status"),
    source_folder: Optional[str] = Query(None, description="Filter by source folder"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=200, description="Items per page")
) -> dict:
    """Get paginated review results with filtering."""
    
    # Verify session exists
    session = db.query(ProcessingSession).filter(ProcessingSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Build query with filters
    query = db.query(ImageResult).filter(ImageResult.session_id == session_id)
    
    if decision:
        query = query.filter(ImageResult.final_decision == decision)
    
    if rejection_reason:
        query = query.filter(ImageResult.rejection_reasons.contains([rejection_reason]))
    
    if human_reviewed is not None:
        if human_reviewed:
            query = query.filter(ImageResult.human_review_at.isnot(None))
        else:
            query = query.filter(ImageResult.human_review_at.is_(None))
    
    if source_folder:
        query = query.filter(ImageResult.source_folder == source_folder)
    
    # Get total count
    total_count = query.count()
    
    # Apply pagination
    offset = (page - 1) * limit
    results = query.offset(offset).limit(limit).all()
    
    # Convert to response format with Thai translations
    response_results = []
    for result in results:
        result_dict = ImageResultResponse.from_orm(result).dict()
        
        # Add Thai translations for rejection reasons
        if result_dict["rejection_reasons"]:
            result_dict["rejection_reasons_thai"] = [
                THAI_REJECTION_REASONS.get(reason, reason) 
                for reason in result_dict["rejection_reasons"]
            ]
        
        response_results.append(result_dict)
    
    return {
        "results": response_results,
        "pagination": {
            "page": page,
            "limit": limit,
            "total_count": total_count,
            "total_pages": (total_count + limit - 1) // limit,
            "has_next": page * limit < total_count,
            "has_prev": page > 1
        },
        "filters": {
            "decision": decision,
            "rejection_reason": rejection_reason,
            "human_reviewed": human_reviewed,
            "source_folder": source_folder
        }
    }

@router.get("/sessions/{session_id}/images/{image_id}/similar")
async def get_similar_images(
    session_id: UUID,
    image_id: UUID,
    db: Session = Depends(get_db)
) -> dict:
    """Get similar images for comparison with enhanced similarity analysis."""
    
    # Get the main image
    main_image = db.query(ImageResult).filter(
        ImageResult.session_id == session_id,
        ImageResult.id == image_id
    ).first()
    
    if not main_image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Get similar images from the same similarity group
    similar_images = []
    if main_image.similarity_group is not None:
        similar_images = db.query(ImageResult).filter(
            ImageResult.session_id == session_id,
            ImageResult.similarity_group == main_image.similarity_group,
            ImageResult.id != image_id
        ).all()
    
    # Also get images from similar_images JSON field with enhanced data
    additional_similar = []
    if main_image.similar_images:
        for similar_data in main_image.similar_images:
            similar_path = similar_data.get("path")
            if similar_path:
                similar_result = db.query(ImageResult).filter(
                    ImageResult.session_id == session_id,
                    ImageResult.image_path == similar_path,
                    ImageResult.id != image_id
                ).first()
                if similar_result:
                    similarity_score = similar_data.get("similarity", 0.0)
                    
                    # Categorize similarity level
                    if similarity_score >= 0.95:
                        similarity_category = "identical"
                        similarity_label = "เหมือนกันทุกประการ"
                    elif similarity_score >= 0.90:
                        similarity_category = "near_duplicate"
                        similarity_label = "เกือบเหมือนกัน"
                    elif similarity_score >= 0.75:
                        similarity_category = "similar"
                        similarity_label = "คล้ายกัน"
                    else:
                        similarity_category = "somewhat_similar"
                        similarity_label = "คล้ายกันบ้าง"
                    
                    additional_similar.append({
                        "image": ImageResultResponse.from_orm(similar_result).dict(),
                        "similarity_score": similarity_score,
                        "similarity_category": similarity_category,
                        "similarity_label": similarity_label,
                        "similarity_percentage": round(similarity_score * 100, 1),
                        "hash_distance": similar_data.get("hash_distance", 0),
                        "feature_distance": similar_data.get("feature_distance", 0.0),
                        "ai_recommendation": similar_data.get("recommendation", "review")
                    })
    
    # Sort by similarity score (highest first)
    additional_similar.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    # Calculate cluster statistics
    all_similarities = [item["similarity_score"] for item in additional_similar]
    cluster_stats = {}
    if all_similarities:
        cluster_stats = {
            "average_similarity": round(np.mean(all_similarities), 3),
            "max_similarity": round(max(all_similarities), 3),
            "min_similarity": round(min(all_similarities), 3),
            "similarity_variance": round(np.var(all_similarities), 3),
            "cluster_quality": "high" if np.mean(all_similarities) > 0.85 else 
                             "medium" if np.mean(all_similarities) > 0.75 else "low"
        }
    
    # Generate AI recommendations for the cluster
    cluster_recommendation = ""
    cluster_recommendation_thai = ""
    if cluster_stats.get("average_similarity", 0) >= 0.95:
        cluster_recommendation = "Keep only the best image, remove others as identical duplicates"
        cluster_recommendation_thai = "เก็บเฉพาะภาพที่ดีที่สุด ลบภาพอื่นที่เหมือนกัน"
    elif cluster_stats.get("average_similarity", 0) >= 0.90:
        cluster_recommendation = "Review carefully - these are near duplicates"
        cluster_recommendation_thai = "ตรวจสอบอย่างละเอียด - ภาพเหล่านี้เกือบเหมือนกัน"
    elif cluster_stats.get("average_similarity", 0) >= 0.75:
        cluster_recommendation = "Similar content - consider keeping variety"
        cluster_recommendation_thai = "เนื้อหาคล้ายกัน - พิจารณาเก็บความหลากหลาย"
    else:
        cluster_recommendation = "Low similarity - likely safe to keep all"
        cluster_recommendation_thai = "ความคล้ายกันต่ำ - น่าจะปลอดภัยที่จะเก็บทั้งหมด"
    
    return {
        "main_image": ImageResultResponse.from_orm(main_image).dict(),
        "similar_images": [ImageResultResponse.from_orm(img).dict() for img in similar_images],
        "additional_similar": additional_similar,
        "total_similar_count": len(similar_images) + len(additional_similar),
        "cluster_stats": cluster_stats,
        "cluster_recommendation": cluster_recommendation,
        "cluster_recommendation_thai": cluster_recommendation_thai,
        "similarity_analysis": {
            "identical_count": len([s for s in additional_similar if s["similarity_category"] == "identical"]),
            "near_duplicate_count": len([s for s in additional_similar if s["similarity_category"] == "near_duplicate"]),
            "similar_count": len([s for s in additional_similar if s["similarity_category"] == "similar"]),
            "somewhat_similar_count": len([s for s in additional_similar if s["similarity_category"] == "somewhat_similar"])
        }
    }

@router.post("/sessions/{session_id}/images/{image_id}/review")
async def review_image(
    session_id: UUID,
    image_id: UUID,
    review_decision: ReviewDecision,
    db: Session = Depends(get_db)
) -> dict:
    """Review a single image (approve or reject)."""
    
    # Get the image result
    image_result = db.query(ImageResult).filter(
        ImageResult.session_id == session_id,
        ImageResult.id == image_id
    ).first()
    
    if not image_result:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Update the decision
    old_decision = image_result.final_decision
    image_result.final_decision = review_decision.decision + "d"  # approved/rejected
    image_result.human_override = True
    image_result.human_review_at = datetime.utcnow()
    
    # Add custom rejection reason if provided
    if review_decision.decision == "reject" and review_decision.reason:
        if not image_result.rejection_reasons:
            image_result.rejection_reasons = []
        if review_decision.reason not in image_result.rejection_reasons:
            image_result.rejection_reasons.append(review_decision.reason)
    
    # Handle file system updates
    output_manager = OutputManager()
    session = db.query(ProcessingSession).filter(ProcessingSession.id == session_id).first()
    
    if review_decision.decision == "approve":
        # Copy to output folder if not already there
        if old_decision == "rejected":
            success = await output_manager.copy_approved_image(
                image_result.image_path,
                session.project.output_folder,
                image_result.source_folder
            )
            if not success:
                raise HTTPException(status_code=500, detail="Failed to copy image to output folder")
    
    elif review_decision.decision == "reject":
        # Remove from output folder if it was previously approved
        if old_decision == "approved":
            await output_manager.remove_rejected_image(
                image_result.image_path,
                session.project.output_folder,
                image_result.source_folder
            )
    
    # Update session statistics
    if old_decision != image_result.final_decision:
        if image_result.final_decision == "approved":
            session.approved_images += 1
            if old_decision == "rejected":
                session.rejected_images -= 1
        elif image_result.final_decision == "rejected":
            session.rejected_images += 1
            if old_decision == "approved":
                session.approved_images -= 1
    
    db.commit()
    
    return {
        "success": True,
        "message": f"Image {review_decision.decision}d successfully",
        "image_id": image_id,
        "new_decision": image_result.final_decision,
        "thai_message": THAI_REJECTION_REASONS.get(
            review_decision.reason, 
            f"ภาพได้รับการ{review_decision.decision}แล้ว"
        ) if review_decision.decision == "reject" else "ภาพได้รับการอนุมัติแล้ว"
    }

@router.post("/sessions/{session_id}/bulk-review")
async def bulk_review_images(
    session_id: UUID,
    bulk_request: BulkReviewRequest,
    db: Session = Depends(get_db)
) -> dict:
    """Bulk review multiple images."""
    
    # Verify session exists
    session = db.query(ProcessingSession).filter(ProcessingSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Convert string IDs to UUIDs
    try:
        uuid_ids = [UUID(image_id) for image_id in bulk_request.image_ids]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid UUID format: {e}")
    
    # Get all images to review
    images = db.query(ImageResult).filter(
        ImageResult.session_id == session_id,
        ImageResult.id.in_(uuid_ids)
    ).all()
    
    if len(images) != len(bulk_request.image_ids):
        raise HTTPException(status_code=404, detail="Some images not found")
    
    # Process each image
    output_manager = OutputManager()
    updated_count = 0
    approved_delta = 0
    rejected_delta = 0
    
    for image_result in images:
        old_decision = image_result.final_decision
        new_decision = bulk_request.decision + "d"  # approved/rejected
        
        # Update the decision
        image_result.final_decision = new_decision
        image_result.human_override = True
        image_result.human_review_at = datetime.utcnow()
        
        # Add custom rejection reason if provided
        if bulk_request.decision == "reject" and bulk_request.reason:
            if not image_result.rejection_reasons:
                image_result.rejection_reasons = []
            if bulk_request.reason not in image_result.rejection_reasons:
                image_result.rejection_reasons.append(bulk_request.reason)
        
        # Handle file system updates
        if bulk_request.decision == "approve":
            if old_decision == "rejected":
                await output_manager.copy_approved_image(
                    image_result.image_path,
                    session.project.output_folder,
                    image_result.source_folder
                )
        elif bulk_request.decision == "reject":
            if old_decision == "approved":
                await output_manager.remove_rejected_image(
                    image_result.image_path,
                    session.project.output_folder,
                    image_result.source_folder
                )
        
        # Track statistics changes
        if old_decision != new_decision:
            if new_decision == "approved":
                approved_delta += 1
                if old_decision == "rejected":
                    rejected_delta -= 1
            elif new_decision == "rejected":
                rejected_delta += 1
                if old_decision == "approved":
                    approved_delta -= 1
        
        updated_count += 1
    
    # Update session statistics
    session.approved_images += approved_delta
    session.rejected_images += rejected_delta
    
    db.commit()
    
    return {
        "success": True,
        "message": f"Bulk {bulk_request.decision} completed",
        "updated_count": updated_count,
        "approved_delta": approved_delta,
        "rejected_delta": rejected_delta,
        "thai_message": f"ดำเนินการ{bulk_request.decision} {updated_count} ภาพเรียบร้อยแล้ว"
    }

@router.get("/sessions/{session_id}/filter-options")
async def get_filter_options(
    session_id: UUID,
    db: Session = Depends(get_db)
) -> dict:
    """Get available filter options for the review interface."""
    
    # Verify session exists
    session = db.query(ProcessingSession).filter(ProcessingSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get all results for this session
    results = db.query(ImageResult).filter(ImageResult.session_id == session_id).all()
    
    # Count by decision
    decision_counts = {}
    for result in results:
        decision = result.final_decision
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
    
    # Count by rejection reasons
    rejection_reason_counts = {}
    for result in results:
        if result.rejection_reasons:
            for reason in result.rejection_reasons:
                rejection_reason_counts[reason] = rejection_reason_counts.get(reason, 0) + 1
    
    # Count by source folder
    folder_counts = {}
    for result in results:
        folder = result.source_folder or "unknown"
        folder_counts[folder] = folder_counts.get(folder, 0) + 1
    
    # Count by human review status
    human_reviewed_count = len([r for r in results if r.human_review_at is not None])
    not_reviewed_count = len(results) - human_reviewed_count
    
    return {
        "decisions": [
            {"value": "all", "label": "ทั้งหมด", "label_en": "All", "count": len(results)},
            {"value": "approved", "label": "ผ่าน", "label_en": "Approved", "count": decision_counts.get("approved", 0)},
            {"value": "rejected", "label": "ไม่ผ่าน", "label_en": "Rejected", "count": decision_counts.get("rejected", 0)},
            {"value": "pending", "label": "รอตรวจสอบ", "label_en": "Pending", "count": decision_counts.get("pending", 0)}
        ],
        "rejection_reasons": [
            {
                "value": reason,
                "label": THAI_REJECTION_REASONS.get(reason, reason),
                "label_en": reason.replace("_", " ").title(),
                "count": count
            }
            for reason, count in rejection_reason_counts.items()
        ],
        "source_folders": [
            {"value": folder, "label": f"โฟลเดอร์ {folder}", "label_en": f"Folder {folder}", "count": count}
            for folder, count in folder_counts.items()
        ],
        "review_status": [
            {"value": "all", "label": "ทั้งหมด", "label_en": "All", "count": len(results)},
            {"value": "reviewed", "label": "ตรวจสอบแล้ว", "label_en": "Reviewed", "count": human_reviewed_count},
            {"value": "not_reviewed", "label": "ยังไม่ตรวจสอบ", "label_en": "Not Reviewed", "count": not_reviewed_count}
        ]
    }

from datetime import datetime

@router.get("/sessions/{session_id}/similarity-config")
async def get_similarity_config(
    session_id: UUID,
    db: Session = Depends(get_db)
) -> dict:
    """Get current similarity detection configuration."""
    
    # Verify session exists
    session = db.query(ProcessingSession).filter(ProcessingSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get current configuration from session metadata or use defaults
    similarity_config = session.config.get('similarity', {}) if session.config else {}
    
    return {
        "current_config": {
            "use_case": similarity_config.get('use_case', 'balanced'),
            "clip_threshold": similarity_config.get('clip_threshold', 0.90),
            "visual_threshold": similarity_config.get('visual_threshold', 0.85),
            "identical_threshold": similarity_config.get('identical_threshold', 0.95),
            "near_duplicate_threshold": similarity_config.get('near_duplicate_threshold', 0.90),
            "similar_threshold": similarity_config.get('similar_threshold', 0.75),
            "clustering_eps": similarity_config.get('clustering_eps', 0.3),
            "min_cluster_size": similarity_config.get('min_cluster_size', 2)
        },
        "use_case_presets": {
            "strict": {
                "description": "เข้มงวดมาก - ตรวจจับภาพที่คล้ายกันเล็กน้อย",
                "description_en": "Very strict - detects even slightly similar images",
                "clip_threshold": 0.95,
                "visual_threshold": 0.90,
                "identical_threshold": 0.98,
                "near_duplicate_threshold": 0.95,
                "similar_threshold": 0.85,
                "clustering_eps": 0.2
            },
            "balanced": {
                "description": "สมดุล - เหมาะสำหรับการใช้งานทั่วไป",
                "description_en": "Balanced - suitable for general use",
                "clip_threshold": 0.90,
                "visual_threshold": 0.85,
                "identical_threshold": 0.95,
                "near_duplicate_threshold": 0.90,
                "similar_threshold": 0.75,
                "clustering_eps": 0.3
            },
            "lenient": {
                "description": "ผ่อนปรน - ตรวจจับเฉพาะภาพที่คล้ายกันมาก",
                "description_en": "Lenient - detects only very similar images",
                "clip_threshold": 0.85,
                "visual_threshold": 0.80,
                "identical_threshold": 0.90,
                "near_duplicate_threshold": 0.85,
                "similar_threshold": 0.70,
                "clustering_eps": 0.4
            }
        },
        "threshold_descriptions": {
            "clip_threshold": "ความคล้ายกันทางความหมาย (CLIP embeddings)",
            "visual_threshold": "ความคล้ายกันทางภาพ (Visual features)",
            "identical_threshold": "ภาพเหมือนกันทุกประการ",
            "near_duplicate_threshold": "ภาพเกือบเหมือนกัน",
            "similar_threshold": "ภาพคล้ายกัน",
            "clustering_eps": "ระยะห่างสำหรับการจัดกลุ่ม",
            "min_cluster_size": "ขนาดกลุ่มขั้นต่ำ"
        }
    }

@router.post("/sessions/{session_id}/similarity-config")
async def update_similarity_config(
    session_id: UUID,
    config_update: dict,
    db: Session = Depends(get_db)
) -> dict:
    """Update similarity detection configuration."""
    
    # Verify session exists
    session = db.query(ProcessingSession).filter(ProcessingSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Validate configuration values
    valid_keys = {
        'use_case', 'clip_threshold', 'visual_threshold', 'identical_threshold',
        'near_duplicate_threshold', 'similar_threshold', 'clustering_eps', 'min_cluster_size'
    }
    
    # Filter valid keys and validate ranges
    validated_config = {}
    for key, value in config_update.items():
        if key in valid_keys:
            if key == 'use_case':
                if value in ['strict', 'balanced', 'lenient']:
                    validated_config[key] = value
            elif key == 'min_cluster_size':
                if isinstance(value, int) and 1 <= value <= 10:
                    validated_config[key] = value
            elif key.endswith('_threshold') or key == 'clustering_eps':
                if isinstance(value, (int, float)) and 0.0 <= value <= 1.0:
                    validated_config[key] = float(value)
    
    # Update session configuration
    if not session.config:
        session.config = {}
    if 'similarity' not in session.config:
        session.config['similarity'] = {}
    
    session.config['similarity'].update(validated_config)
    session.updated_at = datetime.utcnow()
    
    db.commit()
    
    return {
        "success": True,
        "message": "Similarity configuration updated",
        "updated_config": validated_config,
        "thai_message": "อัปเดตการตั้งค่าการตรวจจับความคล้ายกันแล้ว"
    }