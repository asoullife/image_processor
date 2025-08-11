"use client";

import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { CheckCircle, XCircle, Zap, AlertTriangle } from "lucide-react";

interface BulkActionsProps {
  selectedCount: number;
  onApprove: () => void;
  onReject: (reason?: string) => void;
  isLoading: boolean;
}

const REJECTION_REASONS = [
  { value: "low_quality", label: "คุณภาพภาพต่ำ ไม่เหมาะสำหรับขาย" },
  { value: "defects_detected", label: "พบความผิดปกติในภาพ" },
  { value: "similar_images", label: "พบภาพที่คล้ายกัน อาจถูกปฏิเสธเป็น spam" },
  { value: "compliance_issues", label: "ปัญหาลิขสิทธิ์หรือความเป็นส่วนตัว" },
  { value: "technical_issues", label: "ปัญหาทางเทคนิค" },
  { value: "inappropriate_content", label: "เนื้อหาไม่เหมาะสม" },
  { value: "copyright_violation", label: "ละเมิดลิขสิทธิ์" },
  { value: "privacy_violation", label: "ละเมิดความเป็นส่วนตัว" },
];

export function BulkActions({ selectedCount, onApprove, onReject, isLoading }: BulkActionsProps) {
  const [rejectionReason, setRejectionReason] = useState<string>("");

  const handleReject = () => {
    onReject(rejectionReason || undefined);
    setRejectionReason("");
  };

  return (
    <Card className="border-l-4 border-l-primary bg-primary/5">
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Zap className="w-5 h-5 text-primary" />
              <span className="font-medium">Bulk Actions</span>
              <Badge variant="secondary" className="ml-2">
                {selectedCount} images selected
              </Badge>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            {/* Rejection Reason Selector */}
            <div className="flex items-center space-x-2">
              <span className="text-sm text-muted-foreground">Rejection reason:</span>
              <Select value={rejectionReason} onValueChange={setRejectionReason}>
                <SelectTrigger className="w-64">
                  <SelectValue placeholder="Select reason (optional)" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">No specific reason</SelectItem>
                  {REJECTION_REASONS.map((reason) => (
                    <SelectItem key={reason.value} value={reason.value}>
                      {reason.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Action Buttons */}
            <div className="flex items-center space-x-2">
              <Button
                onClick={onApprove}
                disabled={isLoading || selectedCount === 0}
                className="bg-green-600 hover:bg-green-700"
              >
                <CheckCircle className="w-4 h-4 mr-2" />
                อนุมัติทั้งหมด ({selectedCount})
              </Button>
              
              <Button
                variant="destructive"
                onClick={handleReject}
                disabled={isLoading || selectedCount === 0}
              >
                <XCircle className="w-4 h-4 mr-2" />
                ปฏิเสธทั้งหมด ({selectedCount})
              </Button>
            </div>
          </div>
        </div>

        {/* Warning Message */}
        {selectedCount > 10 && (
          <div className="mt-3 p-3 bg-yellow-50 border border-yellow-200 rounded-md flex items-center space-x-2">
            <AlertTriangle className="w-4 h-4 text-yellow-600" />
            <span className="text-sm text-yellow-800">
              You are about to perform bulk action on {selectedCount} images. 
              This will immediately update the output folders. Please confirm your selection.
            </span>
          </div>
        )}

        {/* Loading State */}
        {isLoading && (
          <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-md flex items-center space-x-2">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
            <span className="text-sm text-blue-800">
              Processing bulk action... Please wait.
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}