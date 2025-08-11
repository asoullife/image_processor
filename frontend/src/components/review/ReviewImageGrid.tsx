"use client";

import { useState } from "react";
import Image from "next/image";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { 
  CheckCircle, 
  XCircle, 
  Eye, 
  Clock, 
  User, 
  Bot,
  Zap,
  AlertTriangle,
  Copy
} from "lucide-react";
import { ImageResult } from "@/types";
import { cn } from "@/lib/utils";

interface ReviewImageGridProps {
  images: ImageResult[];
  viewMode: "grid" | "list";
  selectedImages: Set<string>;
  onImageSelect: (imageId: string, selected: boolean) => void;
  onReviewImage: (imageId: string, decision: "approve" | "reject", reason?: string) => void;
  onViewSimilar: (imageId: string) => void;
  isReviewing: boolean;
}

interface ImageCardProps {
  image: ImageResult;
  viewMode: "grid" | "list";
  isSelected: boolean;
  onSelect: (selected: boolean) => void;
  onReview: (decision: "approve" | "reject", reason?: string) => void;
  onViewSimilar: () => void;
  isReviewing: boolean;
}

function ImageCard({ 
  image, 
  viewMode, 
  isSelected, 
  onSelect, 
  onReview, 
  onViewSimilar,
  isReviewing 
}: ImageCardProps) {
  const [showRejectReasons, setShowRejectReasons] = useState(false);

  const getDecisionColor = (decision: string) => {
    switch (decision) {
      case "approved": return "text-green-600 bg-green-50";
      case "rejected": return "text-red-600 bg-red-50";
      case "pending": return "text-yellow-600 bg-yellow-50";
      default: return "text-gray-600 bg-gray-50";
    }
  };

  const getDecisionIcon = (decision: string) => {
    switch (decision) {
      case "approved": return <CheckCircle className="w-4 h-4" />;
      case "rejected": return <XCircle className="w-4 h-4" />;
      case "pending": return <Clock className="w-4 h-4" />;
      default: return <AlertTriangle className="w-4 h-4" />;
    }
  };

  const getDecisionText = (decision: string) => {
    switch (decision) {
      case "approved": return "ผ่าน";
      case "rejected": return "ไม่ผ่าน";
      case "pending": return "รอตรวจสอบ";
      default: return "ไม่ทราบ";
    }
  };

  const thumbnailUrl = `/api/thumbnails/${image.id}`;
  const hasSimilarImages = image.similar_images && image.similar_images.length > 0;

  if (viewMode === "list") {
    return (
      <Card className={cn("transition-all duration-200", isSelected && "ring-2 ring-primary")}>
        <CardContent className="p-4">
          <div className="flex items-center space-x-4">
            {/* Selection Checkbox */}
            <Checkbox
              checked={isSelected}
              onCheckedChange={onSelect}
              className="flex-shrink-0"
            />

            {/* Thumbnail */}
            <div className="relative w-16 h-16 flex-shrink-0">
              <Image
                src={thumbnailUrl}
                alt={image.filename}
                fill
                className="object-cover rounded"
                sizes="64px"
              />
              {hasSimilarImages && (
                <Badge className="absolute -top-1 -right-1 text-xs p-1">
                  <Copy className="w-3 h-3" />
                </Badge>
              )}
            </div>

            {/* Image Info */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center space-x-2 mb-1">
                <h3 className="font-medium truncate">{image.filename}</h3>
                <Badge variant="outline" className="text-xs">
                  โฟลเดอร์ {image.source_folder}
                </Badge>
              </div>
              
              {/* Rejection Reasons */}
              {image.rejection_reasons && image.rejection_reasons.length > 0 && (
                <div className="flex flex-wrap gap-1 mb-2">
                  {image.rejection_reasons.map((reason, index) => (
                    <Badge key={index} variant="destructive" className="text-xs">
                      {(image as any).rejection_reasons_thai?.[index] || reason}
                    </Badge>
                  ))}
                </div>
              )}

              {/* Quality Scores */}
              {image.quality_scores && (
                <div className="flex space-x-4 text-sm text-muted-foreground">
                  <span>คุณภาพ: {(image.quality_scores.overall_score * 100).toFixed(0)}%</span>
                  <span>เวลา: {image.processing_time.toFixed(2)}s</span>
                </div>
              )}
            </div>

            {/* Decision Status */}
            <div className="flex items-center space-x-2">
              <Badge className={cn("flex items-center space-x-1", getDecisionColor(image.final_decision))}>
                {getDecisionIcon(image.final_decision)}
                <span>{getDecisionText(image.final_decision)}</span>
              </Badge>
              
              {image.human_override && (
                <Badge variant="outline" className="flex items-center space-x-1">
                  <User className="w-3 h-3" />
                  <span>มนุษย์</span>
                </Badge>
              )}
            </div>

            {/* Actions */}
            <div className="flex items-center space-x-1">
              {hasSimilarImages && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={onViewSimilar}
                  className="flex items-center space-x-1"
                >
                  <Eye className="w-4 h-4" />
                  <span>ดูภาพคล้าย</span>
                </Button>
              )}
              
              {image.final_decision !== "approved" && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => onReview("approve")}
                  disabled={isReviewing}
                  className="text-green-600 hover:text-green-700"
                >
                  <CheckCircle className="w-4 h-4" />
                </Button>
              )}
              
              {image.final_decision !== "rejected" && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => onReview("reject")}
                  disabled={isReviewing}
                  className="text-red-600 hover:text-red-700"
                >
                  <XCircle className="w-4 h-4" />
                </Button>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Grid view
  return (
    <Card className={cn(
      "group transition-all duration-200 hover:shadow-lg", 
      isSelected && "ring-2 ring-primary"
    )}>
      <CardContent className="p-0">
        {/* Image */}
        <div className="relative aspect-square">
          <Image
            src={thumbnailUrl}
            alt={image.filename}
            fill
            className="object-cover rounded-t-lg"
            sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
          />
          
          {/* Selection Checkbox */}
          <div className="absolute top-2 left-2">
            <Checkbox
              checked={isSelected}
              onCheckedChange={onSelect}
              className="bg-white/80 backdrop-blur-sm"
            />
          </div>

          {/* Similar Images Indicator */}
          {hasSimilarImages && (
            <Badge className="absolute top-2 right-2 bg-blue-500">
              <Copy className="w-3 h-3 mr-1" />
              {image.similar_images.length}
            </Badge>
          )}

          {/* Decision Status */}
          <div className="absolute bottom-2 left-2">
            <Badge className={cn("flex items-center space-x-1", getDecisionColor(image.final_decision))}>
              {getDecisionIcon(image.final_decision)}
              <span>{getDecisionText(image.final_decision)}</span>
            </Badge>
          </div>

          {/* Human Override Indicator */}
          {image.human_override && (
            <Badge variant="outline" className="absolute bottom-2 right-2 bg-white/80">
              <User className="w-3 h-3 mr-1" />
              มนุษย์
            </Badge>
          )}

          {/* Hover Actions */}
          <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-center justify-center space-x-2">
            {hasSimilarImages && (
              <Button
                variant="secondary"
                size="sm"
                onClick={onViewSimilar}
                className="flex items-center space-x-1"
              >
                <Eye className="w-4 h-4" />
                <span>ดูภาพคล้าย</span>
              </Button>
            )}
          </div>
        </div>

        {/* Image Details */}
        <div className="p-3 space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="font-medium text-sm truncate flex-1">{image.filename}</h3>
            <Badge variant="outline" className="text-xs ml-2">
              โฟลเดอร์ {image.source_folder}
            </Badge>
          </div>

          {/* Rejection Reasons */}
          {image.rejection_reasons && image.rejection_reasons.length > 0 && (
            <div className="space-y-1">
              <div className="flex flex-wrap gap-1">
                {image.rejection_reasons.slice(0, 2).map((reason, index) => (
                  <Badge key={index} variant="destructive" className="text-xs">
                    {(image as any).rejection_reasons_thai?.[index] || reason}
                  </Badge>
                ))}
                {image.rejection_reasons.length > 2 && (
                  <Badge variant="outline" className="text-xs">
                    +{image.rejection_reasons.length - 2} more
                  </Badge>
                )}
              </div>
            </div>
          )}

          {/* Quality Info */}
          {image.quality_scores && (
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>คุณภาพ: {(image.quality_scores.overall_score * 100).toFixed(0)}%</span>
              <span>{image.processing_time.toFixed(2)}s</span>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex space-x-1 pt-2">
            {image.final_decision !== "approved" && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => onReview("approve")}
                disabled={isReviewing}
                className="flex-1 text-green-600 hover:text-green-700"
              >
                <CheckCircle className="w-4 h-4 mr-1" />
                อนุมัติ
              </Button>
            )}
            
            {image.final_decision !== "rejected" && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => onReview("reject")}
                disabled={isReviewing}
                className="flex-1 text-red-600 hover:text-red-700"
              >
                <XCircle className="w-4 h-4 mr-1" />
                ปฏิเสธ
              </Button>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function ReviewImageGrid({
  images,
  viewMode,
  selectedImages,
  onImageSelect,
  onReviewImage,
  onViewSimilar,
  isReviewing
}: ReviewImageGridProps) {
  if (images.length === 0) {
    return (
      <Card>
        <CardContent className="p-8 text-center">
          <p className="text-muted-foreground">No images found matching the current filters.</p>
        </CardContent>
      </Card>
    );
  }

  if (viewMode === "list") {
    return (
      <div className="space-y-2">
        {images.map((image) => (
          <ImageCard
            key={image.id}
            image={image}
            viewMode={viewMode}
            isSelected={selectedImages.has(image.id)}
            onSelect={(selected) => onImageSelect(image.id, selected)}
            onReview={(decision, reason) => onReviewImage(image.id, decision, reason)}
            onViewSimilar={() => onViewSimilar(image.id)}
            isReviewing={isReviewing}
          />
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
      {images.map((image) => (
        <ImageCard
          key={image.id}
          image={image}
          viewMode={viewMode}
          isSelected={selectedImages.has(image.id)}
          onSelect={(selected) => onImageSelect(image.id, selected)}
          onReview={(decision, reason) => onReviewImage(image.id, decision, reason)}
          onViewSimilar={() => onViewSimilar(image.id)}
          isReviewing={isReviewing}
        />
      ))}
    </div>
  );
}