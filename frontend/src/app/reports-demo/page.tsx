"use client";

import { ReportsOverview } from "@/components/reports/ReportsOverview";

export default function ReportsDemoPage() {
  // Mock session ID for demo
  const mockSessionId = "demo-session-123";

  return (
    <div className="container mx-auto py-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">Reports Demo</h1>
        <p className="text-muted-foreground">
          Demo page for testing the web-based reports and analytics system
        </p>
      </div>
      
      <ReportsOverview 
        sessionId={mockSessionId}
        projectName="Demo Project"
        onImageSelect={(imageId) => {
          console.log("Selected image:", imageId);
          alert(`Selected image: ${imageId}`);
        }}
      />
    </div>
  );
}