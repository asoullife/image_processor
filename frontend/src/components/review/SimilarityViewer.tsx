"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Image from "next/image";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { 
  CheckCircle, 
  XCircle, 
  Eye, 
  ZoomIn, 
  ZoomOut, 
  RotateCcw,
  ArrowLeft,
  ArrowRight,
  Star
} from "lucide-react";
import { ImageResult } from "@/types";
import { cn } from "@/lib/utils";

interface SimilarityViewerProps {
  sessionId: string;
  imageId: string;
  onClose: () => void;
  onReview: (imageId: string, decision: "approve" | "reject", reason?: string) => void;
}

interface SimilarImageData {
  image: ImageResult;
  similarity_score?: number;
  similarity_category?: string;
  similarity_label?: string;
  similarity_percentage?: number;
  hash_distance?: number;
  feature_distance?: number;
  ai_recommendation?: string;
}

interface ClusterStats {
  average_similarity: number;
  max_similarity: number;
  min_similarity: number;
  similarity_variance: number;
  cluster_quality: string;
}

interface SimilarityAnalysis {
  identical_count: number;
  near_duplicate_count: number;
  similar_count: number;
  somewhat_similar_count: number;
}

export function SimilarityViewer({ 
  sessionId, 
  imageId, 
  onClose, 
  onReview 
}: SimilarityViewerProps) {
  const [selectedImage, setSelectedImage] = useState<string>(imageId);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [viewMode, setViewMode] = useState<'single' | 'comparison'>('single');
  const [comparisonImage, setComparisonImage] = useState<string | null>(null);

  // Fetch similar images
  const { data: similarityData, isLoading } = useQuery({
    queryKey: ["similarity", sessionId, imageId],
    queryFn: async () => {
      const response = await fetch(`/api/review/sessions/${sessionId}/images/${imageId}/similar`);
      return response.json();
    },
  });

  const mainImage = similarityData?.main_image;
  const similarImages = similarityData?.similar_images || [];
  const additionalSimilar = similarityData?.additional_similar || [];
  const clusterStats: ClusterStats = similarityData?.cluster_stats || {};
  const clusterRecommendation = similarityData?.cluster_recommendation_thai || "";
  const similarityAnalysis: SimilarityAnalysis = similarityData?.similarity_analysis || {};
  
  // Combine all similar images
  const allSimilarImages: SimilarImageData[] = [
    ...similarImages.map((img: ImageResult) => ({ image: img })),
    ...additionalSimilar
  ];

  const currentImage = selectedImage === imageId 
    ? mainImage 
    : allSimilarImages.find(s => s.image.id === selectedImage)?.image;

  const handleZoomIn = () => setZoomLevel(prev => Math.min(prev * 1.2, 3));
  const handleZoomOut = () => setZoomLevel(prev => Math.max(prev / 1.2, 0.5));
  const handleResetZoom = () => setZoomLevel(1);

  const getDecisionColor = (decision: string) => {
    switch (decision) {
      case "approved": return "text-green-600 bg-green-50";
      case "rejected": return "text-red-600 bg-red-50";
      case "pending": return "text-yellow-600 bg-yellow-50";
      default: return "text-gray-600 bg-gray-50";
    }
  };

  const getSimilarityColor = (category: string) => {
    switch (category) {
      case "identical": return "text-red-600 bg-red-50 border-red-200";
      case "near_duplicate": return "text-orange-600 bg-orange-50 border-orange-200";
      case "similar": return "text-yellow-600 bg-yellow-50 border-yellow-200";
      case "somewhat_similar": return "text-blue-600 bg-blue-50 border-blue-200";
      default: return "text-gray-600 bg-gray-50 border-gray-200";
    }
  };

  const handleComparisonToggle = (imageId: string) => {
    if (viewMode === 'comparison' && comparisonImage === imageId) {
      setViewMode('single');
      setComparisonImage(null);
    } else {
      setViewMode('comparison');
      setComparisonImage(imageId);
    }
  };

  const handleBulkAction = (action: 'approve_best' | 'reject_duplicates' | 'review_all') => {
    // Implementation for bulk actions based on AI recommendations
    if (action === 'approve_best') {
      // Find the best image (main image or highest quality similar)
      onReview(imageId, "approve");
      // Reject others
      allSimilarImages.forEach(similar => {
        if (similar.ai_recommendation === 'remove_identical' || similar.ai_recommendation === 'review_duplicate') {
          onReview(similar.image.id, "reject", "similar_images");
        }
      });
    }
  };

  if (isLoading) {
    return (
      <Dialog open onOpenChange={onClose}>
        <DialogContent className="max-w-6xl h-[80vh]">
          <div className="flex items-center justify-center h-full">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
            <p className="ml-4">Loading similar images...</p>
          </div>
        </DialogContent>
      </Dialog>
    );
  }

  return (
    <Dialog open onOpenChange={onClose}>
      <DialogContent className="max-w-7xl h-[90vh] p-0">
        <DialogHeader className="p-6 pb-0">
          <DialogTitle className="flex items-center justify-between">
            <div>
              <span>Similarity Comparison - เปรียบเทียบภาพที่คล้าย</span>
              {clusterStats.cluster_quality && (
                <Badge 
                  className={cn(
                    "ml-2",
                    clusterStats.cluster_quality === 'high' ? "bg-red-100 text-red-800" :
                    clusterStats.cluster_quality === 'medium' ? "bg-yellow-100 text-yellow-800" :
                    "bg-green-100 text-green-800"
                  )}
                >
                  {clusterStats.cluster_quality === 'high' ? 'ความคล้ายสูง' :
                   clusterStats.cluster_quality === 'medium' ? 'ความคล้ายปานกลาง' :
                   'ความคล้ายต่ำ'}
                </Badge>
              )}
            </div>
            <div className="flex items-center space-x-2">
              <Badge variant="outline">
                {allSimilarImages.length + 1} images total
              </Badge>
              <Button 
                variant="outline" 
                size="sm" 
                onClick={() => setViewMode(viewMode === 'single' ? 'comparison' : 'single')}
              >
                {viewMode === 'single' ? 'Compare' : 'Single'}
              </Button>
              <Button variant="outline" size="sm" onClick={onClose}>
                Close
              </Button>
            </div>
          </DialogTitle>
          
          {/* Cluster Statistics */}
          {clusterStats.average_similarity && (
            <div className="mt-4 p-4 bg-muted/30 rounded-lg">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-muted-foreground">Average Similarity:</span>
                  <span className="ml-2 font-medium">{(clusterStats.average_similarity * 100).toFixed(1)}%</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Max Similarity:</span>
                  <span className="ml-2 font-medium">{(clusterStats.max_similarity * 100).toFixed(1)}%</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Identical:</span>
                  <span className="ml-2 font-medium text-red-600">{similarityAnalysis.identical_count}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Near Duplicates:</span>
                  <span className="ml-2 font-medium text-orange-600">{similarityAnalysis.near_duplicate_count}</span>
                </div>
              </div>
              
              {clusterRecommendation && (
                <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded">
                  <div className="flex items-start space-x-2">
                    <div className="w-5 h-5 rounded-full bg-blue-500 flex items-center justify-center flex-shrink-0 mt-0.5">
                      <span className="text-white text-xs">AI</span>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-blue-900">AI Recommendation:</p>
                      <p className="text-sm text-blue-800">{clusterRecommendation}</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </DialogHeader>

        <div className="flex h-full">
          {/* Sidebar - Similar Images List */}
          <div className="w-80 border-r bg-muted/30">
            <ScrollArea className="h-full p-4">
              <div className="space-y-3">
                {/* Main Image */}
                <Card 
                  className={cn(
                    "cursor-pointer transition-all",
                    selectedImage === imageId && "ring-2 ring-primary"
                  )}
                  onClick={() => setSelectedImage(imageId)}
                >
                  <CardContent className="p-3">
                    <div className="flex items-center space-x-3">
                      <div className="relative w-16 h-16 flex-shrink-0">
                        <Image
                          src={`/api/thumbnails/${imageId}`}
                          alt={mainImage?.filename || "Main image"}
                          fill
                          className="object-cover rounded"
                          sizes="64px"
                        />
                        <Badge className="absolute -top-1 -right-1 bg-blue-500 text-xs p-1">
                          <Star className="w-3 h-3" />
                        </Badge>
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="font-medium text-sm truncate">
                          {mainImage?.filename}
                        </p>
                        <p className="text-xs text-muted-foreground">Main Image</p>
                        <Badge className={cn("text-xs mt-1", getDecisionColor(mainImage?.final_decision || ""))}>
                          {mainImage?.final_decision === "approved" ? "ผ่าน" : 
                           mainImage?.final_decision === "rejected" ? "ไม่ผ่าน" : "รอตรวจสอบ"}
                        </Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Similar Images */}
                {allSimilarImages.map((similarData, index) => {
                  const image = similarData.image;
                  const similarity = similarData.similarity_score;
                  const category = similarData.similarity_category;
                  const label = similarData.similarity_label;
                  const percentage = similarData.similarity_percentage;
                  
                  return (
                    <Card 
                      key={image.id}
                      className={cn(
                        "cursor-pointer transition-all",
                        selectedImage === image.id && "ring-2 ring-primary",
                        category && getSimilarityColor(category)
                      )}
                      onClick={() => setSelectedImage(image.id)}
                    >
                      <CardContent className="p-3">
                        <div className="flex items-center space-x-3">
                          <div className="relative w-16 h-16 flex-shrink-0">
                            <Image
                              src={`/api/thumbnails/${image.id}`}
                              alt={image.filename}
                              fill
                              className="object-cover rounded"
                              sizes="64px"
                            />
                            {/* Similarity indicator */}
                            {category && (
                              <div className={cn(
                                "absolute -top-1 -right-1 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold",
                                category === "identical" ? "bg-red-500 text-white" :
                                category === "near_duplicate" ? "bg-orange-500 text-white" :
                                category === "similar" ? "bg-yellow-500 text-white" :
                                "bg-blue-500 text-white"
                              )}>
                                {percentage ? Math.round(percentage) : '?'}
                              </div>
                            )}
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="font-medium text-sm truncate">
                              {image.filename}
                            </p>
                            <div className="flex items-center space-x-2 mt-1">
                              {similarity && (
                                <Badge variant="outline" className="text-xs">
                                  {(similarity * 100).toFixed(1)}%
                                </Badge>
                              )}
                              {label && (
                                <Badge className={cn("text-xs", getSimilarityColor(category || ""))}>
                                  {label}
                                </Badge>
                              )}
                            </div>
                            <Badge className={cn("text-xs mt-1", getDecisionColor(image.final_decision))}>
                              {image.final_decision === "approved" ? "ผ่าน" : 
                               image.final_decision === "rejected" ? "ไม่ผ่าน" : "รอตรวจสอบ"}
                            </Badge>
                          </div>
                          <div className="flex flex-col space-y-1">
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-6 w-6 p-0"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleComparisonToggle(image.id);
                              }}
                            >
                              <Eye className="w-3 h-3" />
                            </Button>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            </ScrollArea>
          </div>

          {/* Main Viewer */}
          <div className="flex-1 flex flex-col">
            {/* Toolbar */}
            <div className="flex items-center justify-between p-4 border-b bg-muted/30">
              <div className="flex items-center space-x-2">
                <h3 className="font-medium">{currentImage?.filename}</h3>
                <Badge variant="outline">
                  โฟลเดอร์ {currentImage?.source_folder}
                </Badge>
                {currentImage?.human_override && (
                  <Badge variant="outline">Human Reviewed</Badge>
                )}
              </div>
              
              <div className="flex items-center space-x-2">
                <Button variant="outline" size="sm" onClick={handleZoomOut}>
                  <ZoomOut className="w-4 h-4" />
                </Button>
                <span className="text-sm text-muted-foreground min-w-16 text-center">
                  {Math.round(zoomLevel * 100)}%
                </span>
                <Button variant="outline" size="sm" onClick={handleZoomIn}>
                  <ZoomIn className="w-4 h-4" />
                </Button>
                <Button variant="outline" size="sm" onClick={handleResetZoom}>
                  <RotateCcw className="w-4 h-4" />
                </Button>
              </div>
            </div>

            {/* Image Display */}
            <div className="flex-1 overflow-auto bg-gray-100 flex items-center justify-center p-4">
              {viewMode === 'single' ? (
                // Single image view
                currentImage && (
                  <div 
                    className="relative transition-transform duration-200"
                    style={{ transform: `scale(${zoomLevel})` }}
                  >
                    <Image
                      src={`/api/images/${currentImage.id}/full`}
                      alt={currentImage.filename}
                      width={800}
                      height={600}
                      className="max-w-full max-h-full object-contain shadow-lg"
                      priority
                    />
                  </div>
                )
              ) : (
                // Side-by-side comparison view
                <div className="flex space-x-4 w-full h-full">
                  {/* Main image */}
                  <div className="flex-1 flex flex-col">
                    <div className="text-center mb-2">
                      <Badge variant="outline">Main Image</Badge>
                    </div>
                    {mainImage && (
                      <div 
                        className="relative flex-1 flex items-center justify-center transition-transform duration-200"
                        style={{ transform: `scale(${zoomLevel})` }}
                      >
                        <Image
                          src={`/api/images/${mainImage.id}/full`}
                          alt={mainImage.filename}
                          width={400}
                          height={400}
                          className="max-w-full max-h-full object-contain shadow-lg"
                          priority
                        />
                      </div>
                    )}
                  </div>
                  
                  {/* Comparison image */}
                  <div className="flex-1 flex flex-col">
                    <div className="text-center mb-2">
                      <Badge variant="outline">Comparison</Badge>
                      {comparisonImage && (() => {
                        const compData = allSimilarImages.find(s => s.image.id === comparisonImage);
                        return compData?.similarity_percentage && (
                          <Badge className="ml-2 bg-blue-100 text-blue-800">
                            {compData.similarity_percentage}% similar
                          </Badge>
                        );
                      })()}
                    </div>
                    {comparisonImage && (() => {
                      const compImage = allSimilarImages.find(s => s.image.id === comparisonImage)?.image;
                      return compImage && (
                        <div 
                          className="relative flex-1 flex items-center justify-center transition-transform duration-200"
                          style={{ transform: `scale(${zoomLevel})` }}
                        >
                          <Image
                            src={`/api/images/${compImage.id}/full`}
                            alt={compImage.filename}
                            width={400}
                            height={400}
                            className="max-w-full max-h-full object-contain shadow-lg"
                            priority
                          />
                        </div>
                      );
                    })()}
                  </div>
                </div>
              )}
            </div>

            {/* Image Details & Actions */}
            <div className="p-4 border-t bg-muted/30">
              {currentImage && (
                <div className="space-y-3">
                  {/* Image Info */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <span className="text-muted-foreground">Decision:</span>
                      <Badge className={cn("ml-2", getDecisionColor(currentImage.final_decision))}>
                        {currentImage.final_decision === "approved" ? "ผ่าน" : 
                         currentImage.final_decision === "rejected" ? "ไม่ผ่าน" : "รอตรวจสอบ"}
                      </Badge>
                    </div>
                    {currentImage.quality_scores && (
                      <div>
                        <span className="text-muted-foreground">Quality:</span>
                        <span className="ml-2 font-medium">
                          {(currentImage.quality_scores.overall_score * 100).toFixed(0)}%
                        </span>
                      </div>
                    )}
                    <div>
                      <span className="text-muted-foreground">Processing:</span>
                      <span className="ml-2 font-medium">
                        {currentImage.processing_time.toFixed(2)}s
                      </span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Source:</span>
                      <span className="ml-2 font-medium">
                        โฟลเดอร์ {currentImage.source_folder}
                      </span>
                    </div>
                  </div>

                  {/* Rejection Reasons */}
                  {currentImage.rejection_reasons && currentImage.rejection_reasons.length > 0 && (
                    <div>
                      <span className="text-sm text-muted-foreground mb-2 block">Rejection Reasons:</span>
                      <div className="flex flex-wrap gap-2">
                        {currentImage.rejection_reasons.map((reason, index) => (
                          <Badge key={index} variant="destructive">
                            {(currentImage as any).rejection_reasons_thai?.[index] || reason}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Action Buttons */}
                  <div className="flex items-center justify-between">
                    <div className="flex space-x-2">
                      {currentImage.final_decision !== "approved" && (
                        <Button
                          onClick={() => onReview(currentImage.id, "approve")}
                          className="bg-green-600 hover:bg-green-700"
                        >
                          <CheckCircle className="w-4 h-4 mr-2" />
                          อนุมัติภาพนี้
                        </Button>
                      )}
                      
                      {currentImage.final_decision !== "rejected" && (
                        <Button
                          variant="destructive"
                          onClick={() => onReview(currentImage.id, "reject")}
                        >
                          <XCircle className="w-4 h-4 mr-2" />
                          ปฏิเสธภาพนี้
                        </Button>
                      )}
                    </div>

                    <div className="text-sm text-muted-foreground">
                      {selectedImage === imageId ? "Main Image" : 
                       `Similar Image ${allSimilarImages.findIndex(s => s.image.id === selectedImage) + 1}`}
                    </div>
                  </div>
                  
                  {/* AI-Based Bulk Actions */}
                  {allSimilarImages.length > 0 && (
                    <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                      <h4 className="font-medium text-blue-900 mb-3">AI-Powered Bulk Actions</h4>
                      <div className="flex flex-wrap gap-2">
                        {clusterStats.average_similarity >= 0.95 && (
                          <Button
                            size="sm"
                            onClick={() => handleBulkAction('approve_best')}
                            className="bg-blue-600 hover:bg-blue-700"
                          >
                            <Star className="w-4 h-4 mr-2" />
                            เก็บภาพดีที่สุด ลบที่เหมือนกัน
                          </Button>
                        )}
                        
                        {similarityAnalysis.near_duplicate_count > 0 && (
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => handleBulkAction('reject_duplicates')}
                          >
                            <XCircle className="w-4 h-4 mr-2" />
                            ปฏิเสธภาพซ้ำ ({similarityAnalysis.near_duplicate_count})
                          </Button>
                        )}
                        
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleBulkAction('review_all')}
                        >
                          <Eye className="w-4 h-4 mr-2" />
                          ทำเครื่องหมายให้ตรวจสอบทั้งหมด
                        </Button>
                      </div>
                      
                      <div className="mt-2 text-xs text-blue-700">
                        AI แนะนำการดำเนินการตามระดับความคล้ายกันที่ตรวจพบ
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}