"use client";

import { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { toast } from "sonner";
import { 
  Filter, 
  Search, 
  Eye, 
  CheckCircle, 
  XCircle, 
  Grid3X3, 
  List, 
  ArrowLeft,
  Download,
  RefreshCw,
  Zap
} from "lucide-react";

import { apiClient, queryKeys } from "@/lib/api";
import { ImageResult } from "@/types";
import { ReviewImageGrid } from "@/components/review/ReviewImageGrid";
import { ReviewStats } from "@/components/review/ReviewStats";
import { BulkActions } from "@/components/review/BulkActions";
import { SimilarityViewer } from "@/components/review/SimilarityViewer";
import { SimilarityConfig } from "@/components/review/SimilarityConfig";
import { ReviewFilters } from "@/components/review/ReviewFilters";

interface ReviewFilters {
  decision?: "approved" | "rejected" | "pending";
  rejection_reason?: string;
  human_reviewed?: boolean;
  source_folder?: string;
  search?: string;
}

interface FilterOptions {
  decisions: Array<{ value: string; label: string; label_en: string; count: number }>;
  rejection_reasons: Array<{ value: string; label: string; label_en: string; count: number }>;
  source_folders: Array<{ value: string; label: string; label_en: string; count: number }>;
  review_status: Array<{ value: string; label: string; label_en: string; count: number }>;
}

export default function ReviewPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const queryClient = useQueryClient();
  const projectId = params?.id as string;

  // State
  const [filters, setFilters] = useState<ReviewFilters>({});
  const [selectedImages, setSelectedImages] = useState<Set<string>>(new Set());
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [showSimilarity, setShowSimilarity] = useState<string | null>(null);
  const [showSimilarityConfig, setShowSimilarityConfig] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize] = useState(50);

  // Get current session for this project
  const { data: project } = useQuery({
    queryKey: queryKeys.project(projectId),
    queryFn: () => apiClient.getProject(projectId),
  });

  const sessionId = project?.current_session_id;

  // Get filter options
  const { data: filterOptions } = useQuery<FilterOptions>({
    queryKey: ["review", sessionId, "filter-options"],
    queryFn: async () => {
      const response = await fetch(`/api/review/sessions/${sessionId}/filter-options`);
      return response.json();
    },
    enabled: !!sessionId,
  });

  // Get review results
  const { data: reviewData, isLoading, refetch } = useQuery({
    queryKey: ["review", sessionId, "results", filters, currentPage],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (filters.decision) params.append("decision", filters.decision);
      if (filters.rejection_reason) params.append("rejection_reason", filters.rejection_reason);
      if (filters.human_reviewed !== undefined) params.append("human_reviewed", filters.human_reviewed.toString());
      if (filters.source_folder) params.append("source_folder", filters.source_folder);
      params.append("page", currentPage.toString());
      params.append("limit", pageSize.toString());

      const response = await fetch(`/api/review/sessions/${sessionId}/results?${params}`);
      return response.json();
    },
    enabled: !!sessionId,
  });

  // Review mutations
  const reviewMutation = useMutation({
    mutationFn: async ({ imageId, decision, reason }: { imageId: string; decision: "approve" | "reject"; reason?: string }) => {
      const response = await fetch(`/api/review/sessions/${sessionId}/images/${imageId}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ decision, reason }),
      });
      return response.json();
    },
    onSuccess: (data) => {
      toast.success(data.thai_message || "Review completed successfully");
      queryClient.invalidateQueries({ queryKey: ["review", sessionId] });
      queryClient.invalidateQueries({ queryKey: queryKeys.project(projectId) });
    },
    onError: (error) => {
      toast.error("Failed to review image");
      console.error("Review error:", error);
    },
  });

  const bulkReviewMutation = useMutation({
    mutationFn: async ({ imageIds, decision, reason }: { imageIds: string[]; decision: "approve" | "reject"; reason?: string }) => {
      const response = await fetch(`/api/review/sessions/${sessionId}/bulk-review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_ids: imageIds, decision, reason }),
      });
      return response.json();
    },
    onSuccess: (data) => {
      toast.success(data.thai_message || "Bulk review completed successfully");
      setSelectedImages(new Set());
      queryClient.invalidateQueries({ queryKey: ["review", sessionId] });
      queryClient.invalidateQueries({ queryKey: queryKeys.project(projectId) });
    },
    onError: (error) => {
      toast.error("Failed to perform bulk review");
      console.error("Bulk review error:", error);
    },
  });

  // Handlers
  const handleFilterChange = (newFilters: Partial<ReviewFilters>) => {
    setFilters(prev => ({ ...prev, ...newFilters }));
    setCurrentPage(1);
  };

  const handleImageSelect = (imageId: string, selected: boolean) => {
    const newSelected = new Set(selectedImages);
    if (selected) {
      newSelected.add(imageId);
    } else {
      newSelected.delete(imageId);
    }
    setSelectedImages(newSelected);
  };

  const handleSelectAll = (images: ImageResult[]) => {
    const allIds = new Set(images.map(img => img.id));
    setSelectedImages(allIds);
  };

  const handleClearSelection = () => {
    setSelectedImages(new Set());
  };

  const handleReviewImage = (imageId: string, decision: "approve" | "reject", reason?: string) => {
    reviewMutation.mutate({ imageId, decision, reason });
  };

  const handleBulkReview = (decision: "approve" | "reject", reason?: string) => {
    if (selectedImages.size === 0) {
      toast.error("Please select images to review");
      return;
    }
    bulkReviewMutation.mutate({ 
      imageIds: Array.from(selectedImages), 
      decision, 
      reason 
    });
  };

  const handleViewSimilar = async (imageId: string) => {
    setShowSimilarity(imageId);
  };

  if (!sessionId) {
    return (
      <div className="container mx-auto py-8">
        <Card>
          <CardContent className="p-8 text-center">
            <p className="text-muted-foreground">No active session found for this project.</p>
            <Button onClick={() => router.back()} className="mt-4">
              <ArrowLeft className="w-4 h-4 mr-2" />
              Go Back
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="container mx-auto py-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Button variant="ghost" onClick={() => router.back()}>
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Project
          </Button>
          <div>
            <h1 className="text-2xl font-bold">Human Review System</h1>
            <p className="text-muted-foreground">ระบบตรวจสอบภาพโดยมนุษย์</p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" onClick={() => setShowSimilarityConfig(true)}>
            <Zap className="w-4 h-4 mr-2" />
            Similarity Settings
          </Button>
          <Button variant="outline" onClick={() => refetch()}>
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
          <Button variant="outline">
            <Download className="w-4 h-4 mr-2" />
            Export Results
          </Button>
        </div>
      </div>

      {/* Stats */}
      {reviewData && (
        <ReviewStats 
          totalImages={reviewData.pagination.total_count}
          approvedCount={filterOptions?.decisions.find(d => d.value === "approved")?.count || 0}
          rejectedCount={filterOptions?.decisions.find(d => d.value === "rejected")?.count || 0}
          pendingCount={filterOptions?.decisions.find(d => d.value === "pending")?.count || 0}
          humanReviewedCount={filterOptions?.review_status.find(s => s.value === "reviewed")?.count || 0}
        />
      )}

      {/* Filters and Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Filter className="w-5 h-5 mr-2" />
            Filters & Controls
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Filter Tabs */}
            <Tabs value={filters.decision || "all"} onValueChange={(value) => 
              handleFilterChange({ decision: value === "all" ? undefined : value as any })
            }>
              <TabsList className="grid w-full grid-cols-4">
                {filterOptions?.decisions.map((option) => (
                  <TabsTrigger key={option.value} value={option.value} className="flex items-center space-x-2">
                    <span>{option.label}</span>
                    <Badge variant="secondary" className="text-xs">
                      {option.count}
                    </Badge>
                  </TabsTrigger>
                ))}
              </TabsList>
            </Tabs>

            {/* Additional Filters */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              {/* Rejection Reason Filter */}
              <Select 
                value={filters.rejection_reason || ""} 
                onValueChange={(value) => handleFilterChange({ rejection_reason: value || undefined })}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Filter by rejection reason" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">All Reasons</SelectItem>
                  {filterOptions?.rejection_reasons.map((reason) => (
                    <SelectItem key={reason.value} value={reason.value}>
                      {reason.label} ({reason.count})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              {/* Source Folder Filter */}
              <Select 
                value={filters.source_folder || ""} 
                onValueChange={(value) => handleFilterChange({ source_folder: value || undefined })}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Filter by folder" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">All Folders</SelectItem>
                  {filterOptions?.source_folders.map((folder) => (
                    <SelectItem key={folder.value} value={folder.value}>
                      {folder.label} ({folder.count})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              {/* Human Review Status */}
              <Select 
                value={filters.human_reviewed === undefined ? "" : filters.human_reviewed.toString()} 
                onValueChange={(value) => handleFilterChange({ 
                  human_reviewed: value === "" ? undefined : value === "true" 
                })}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Review status" />
                </SelectTrigger>
                <SelectContent>
                  {filterOptions?.review_status.map((status) => (
                    <SelectItem key={status.value} value={status.value === "all" ? "" : status.value === "reviewed" ? "true" : "false"}>
                      {status.label} ({status.count})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              {/* Search */}
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
                <Input
                  placeholder="Search filenames..."
                  value={filters.search || ""}
                  onChange={(e) => handleFilterChange({ search: e.target.value || undefined })}
                  className="pl-10"
                />
              </div>
            </div>

            {/* View Controls */}
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Button
                  variant={viewMode === "grid" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setViewMode("grid")}
                >
                  <Grid3X3 className="w-4 h-4" />
                </Button>
                <Button
                  variant={viewMode === "list" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setViewMode("list")}
                >
                  <List className="w-4 h-4" />
                </Button>
              </div>

              {/* Selection Controls */}
              {reviewData?.results && (
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-muted-foreground">
                    {selectedImages.size} selected
                  </span>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleSelectAll(reviewData.results)}
                  >
                    Select All
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleClearSelection}
                    disabled={selectedImages.size === 0}
                  >
                    Clear
                  </Button>
                </div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Bulk Actions */}
      {selectedImages.size > 0 && (
        <BulkActions
          selectedCount={selectedImages.size}
          onApprove={() => handleBulkReview("approve")}
          onReject={(reason) => handleBulkReview("reject", reason)}
          isLoading={bulkReviewMutation.isPending}
        />
      )}

      {/* Results */}
      {isLoading ? (
        <Card>
          <CardContent className="p-8 text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto"></div>
            <p className="mt-4 text-muted-foreground">Loading review results...</p>
          </CardContent>
        </Card>
      ) : reviewData?.results ? (
        <ReviewImageGrid
          images={reviewData.results}
          viewMode={viewMode}
          selectedImages={selectedImages}
          onImageSelect={handleImageSelect}
          onReviewImage={handleReviewImage}
          onViewSimilar={handleViewSimilar}
          isReviewing={reviewMutation.isPending}
        />
      ) : (
        <Card>
          <CardContent className="p-8 text-center">
            <p className="text-muted-foreground">No images found matching the current filters.</p>
          </CardContent>
        </Card>
      )}

      {/* Pagination */}
      {reviewData?.pagination && reviewData.pagination.total_pages > 1 && (
        <div className="flex items-center justify-center space-x-2">
          <Button
            variant="outline"
            onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
            disabled={!reviewData.pagination.has_prev}
          >
            Previous
          </Button>
          <span className="text-sm text-muted-foreground">
            Page {reviewData.pagination.page} of {reviewData.pagination.total_pages}
          </span>
          <Button
            variant="outline"
            onClick={() => setCurrentPage(prev => prev + 1)}
            disabled={!reviewData.pagination.has_next}
          >
            Next
          </Button>
        </div>
      )}

      {/* Similarity Viewer Modal */}
      {showSimilarity && (
        <SimilarityViewer
          sessionId={sessionId}
          imageId={showSimilarity}
          onClose={() => setShowSimilarity(null)}
          onReview={handleReviewImage}
        />
      )}

      {/* Similarity Configuration Modal */}
      {showSimilarityConfig && (
        <SimilarityConfig
          sessionId={sessionId}
          onClose={() => setShowSimilarityConfig(false)}
        />
      )}
    </div>
  );
}