"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { CheckCircle, XCircle, Clock, User, TrendingUp } from "lucide-react";

interface ReviewStatsProps {
  totalImages: number;
  approvedCount: number;
  rejectedCount: number;
  pendingCount: number;
  humanReviewedCount: number;
}

export function ReviewStats({
  totalImages,
  approvedCount,
  rejectedCount,
  pendingCount,
  humanReviewedCount
}: ReviewStatsProps) {
  const approvalRate = totalImages > 0 ? (approvedCount / totalImages) * 100 : 0;
  const rejectionRate = totalImages > 0 ? (rejectedCount / totalImages) * 100 : 0;
  const humanReviewRate = totalImages > 0 ? (humanReviewedCount / totalImages) * 100 : 0;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {/* Total Images */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Total Images</CardTitle>
          <TrendingUp className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{totalImages.toLocaleString()}</div>
          <p className="text-xs text-muted-foreground">
            ภาพทั้งหมดในระบบ
          </p>
        </CardContent>
      </Card>

      {/* Approved Images */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Approved</CardTitle>
          <CheckCircle className="h-4 w-4 text-green-600" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold text-green-600">
            {approvedCount.toLocaleString()}
          </div>
          <div className="flex items-center space-x-2">
            <Progress value={approvalRate} className="flex-1 h-2" />
            <span className="text-xs text-muted-foreground">
              {approvalRate.toFixed(1)}%
            </span>
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            ภาพที่ผ่านการอนุมัติ
          </p>
        </CardContent>
      </Card>

      {/* Rejected Images */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Rejected</CardTitle>
          <XCircle className="h-4 w-4 text-red-600" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold text-red-600">
            {rejectedCount.toLocaleString()}
          </div>
          <div className="flex items-center space-x-2">
            <Progress value={rejectionRate} className="flex-1 h-2" />
            <span className="text-xs text-muted-foreground">
              {rejectionRate.toFixed(1)}%
            </span>
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            ภาพที่ถูกปฏิเสธ
          </p>
        </CardContent>
      </Card>

      {/* Human Reviewed */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Human Reviewed</CardTitle>
          <User className="h-4 w-4 text-blue-600" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold text-blue-600">
            {humanReviewedCount.toLocaleString()}
          </div>
          <div className="flex items-center space-x-2">
            <Progress value={humanReviewRate} className="flex-1 h-2" />
            <span className="text-xs text-muted-foreground">
              {humanReviewRate.toFixed(1)}%
            </span>
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            ตรวจสอบโดยมนุษย์แล้ว
          </p>
        </CardContent>
      </Card>

      {/* Pending Review */}
      {pendingCount > 0 && (
        <Card className="md:col-span-2 lg:col-span-4">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Clock className="h-5 w-5 text-yellow-600" />
              <span>Pending Review</span>
              <Badge variant="secondary">{pendingCount} images</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <p className="text-sm text-muted-foreground">
                {pendingCount.toLocaleString()} images are waiting for human review
              </p>
              <Badge variant="outline" className="text-yellow-600">
                {((pendingCount / totalImages) * 100).toFixed(1)}% pending
              </Badge>
            </div>
            <Progress 
              value={(pendingCount / totalImages) * 100} 
              className="mt-2 h-2" 
            />
          </CardContent>
        </Card>
      )}
    </div>
  );
}