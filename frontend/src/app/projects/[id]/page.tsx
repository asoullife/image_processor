"use client";

import { motion } from "framer-motion";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { 
  ArrowLeftIcon,
  PlayIcon,
  PauseIcon,
  SettingsIcon,
  EyeIcon,
  DownloadIcon
} from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { AnimatedCounter, AnimatedProgress } from "@/components/ui/animated-counter";
import { MagicCard } from "@/components/ui/magic-card";
import { SingleProjectHistory } from "@/components/projects/ProjectHistory";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

// Mock data - will be replaced with real data from API
const mockProject = {
  id: "1",
  name: "Wedding Photos Batch 1",
  status: "processing",
  totalImages: 1250,
  processedImages: 850,
  approvedImages: 780,
  rejectedImages: 70,
  createdAt: "2024-01-15T10:30:00Z",
  inputFolder: "/Users/photographer/wedding-batch-1",
  outputFolder: "/Users/photographer/wedding-batch-1_processed",
  performanceMode: "balanced",
  currentImage: "IMG_5432.jpg",
  processingSpeed: 2.3, // images per second
  estimatedCompletion: "2024-01-15T14:30:00Z"
};

const getStatusBadge = (status: string) => {
  switch (status) {
    case "completed":
      return <Badge variant="success">Completed</Badge>;
    case "processing":
      return <Badge variant="default">Processing</Badge>;
    case "paused":
      return <Badge variant="warning">Paused</Badge>;
    case "failed":
      return <Badge variant="destructive">Failed</Badge>;
    default:
      return <Badge variant="secondary">Unknown</Badge>;
  }
};

export default function ProjectDetailPage() {
  const params = useParams<{ id: string }>();
  const projectId = params?.id as string;

  const progressPercentage = (mockProject.processedImages / mockProject.totalImages) * 100;
  const approvalRate = mockProject.processedImages > 0 
    ? (mockProject.approvedImages / mockProject.processedImages) * 100 
    : 0;

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="flex items-center justify-between mb-8"
      >
        <div className="flex items-center gap-4">
          <Button variant="outline" size="sm" asChild>
            <Link href="/projects">
              <ArrowLeftIcon className="w-4 h-4 mr-2" />
              Back to Projects
            </Link>
          </Button>
          <div>
            <div className="flex items-center gap-3 mb-2">
              <h1 className="text-3xl font-bold">{mockProject.name}</h1>
              {getStatusBadge(mockProject.status)}
            </div>
            <p className="text-muted-foreground">
              Project ID: {projectId} • Created {new Date(mockProject.createdAt).toLocaleDateString()}
            </p>
          </div>
        </div>
        
        <div className="flex gap-2">
          {mockProject.status === "processing" && (
            <Button variant="outline">
              <PauseIcon className="w-4 h-4 mr-2" />
              Pause
            </Button>
          )}
          {mockProject.status === "paused" && (
            <Button variant="outline">
              <PlayIcon className="w-4 h-4 mr-2" />
              Resume
            </Button>
          )}
          <Button variant="outline">
            <EyeIcon className="w-4 h-4 mr-2" />
            Review Results
          </Button>
          <Button variant="outline">
            <SettingsIcon className="w-4 h-4 mr-2" />
            Settings
          </Button>
        </div>
      </motion.div>

      {/* Progress Overview */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8"
      >
        {/* Main Progress */}
        <MagicCard className="lg:col-span-2" glowEffect={mockProject.status === "processing"}>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              Processing Progress
              <span className="text-sm font-normal text-muted-foreground">
                <AnimatedCounter value={mockProject.processedImages} /> / <AnimatedCounter value={mockProject.totalImages} />
              </span>
            </CardTitle>
            <CardDescription>
              {mockProject.status === "processing" && (
                <>Currently processing: {mockProject.currentImage}</>
              )}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <AnimatedProgress 
              value={mockProject.processedImages} 
              max={mockProject.totalImages}
              showPercentage
              className="mb-4"
            />
            
            {mockProject.status === "processing" && (
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-muted-foreground">Speed:</span>
                  <span className="ml-2 font-medium">
                    <AnimatedCounter value={mockProject.processingSpeed} suffix=" img/sec" />
                  </span>
                </div>
                <div>
                  <span className="text-muted-foreground">ETA:</span>
                  <span className="ml-2 font-medium">
                    {new Date(mockProject.estimatedCompletion).toLocaleTimeString()}
                  </span>
                </div>
              </div>
            )}
          </CardContent>
        </MagicCard>

        {/* Quick Stats */}
        <MagicCard>
          <CardHeader>
            <CardTitle>Results Summary</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-muted-foreground">Approved</span>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-500 rounded-full" />
                <span className="font-medium text-green-600">
                  <AnimatedCounter value={mockProject.approvedImages} />
                </span>
              </div>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-muted-foreground">Rejected</span>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-red-500 rounded-full" />
                <span className="font-medium text-red-600">
                  <AnimatedCounter value={mockProject.rejectedImages} />
                </span>
              </div>
            </div>
            <div className="pt-2 border-t">
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Approval Rate</span>
                <span className="font-medium">
                  <AnimatedCounter value={approvalRate} suffix="%" />
                </span>
              </div>
            </div>
          </CardContent>
        </MagicCard>
      </motion.div>

      {/* Project Details */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8"
      >
        <Card>
          <CardHeader>
            <CardTitle>Project Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Performance Mode:</span>
              <Badge variant="outline">{mockProject.performanceMode}</Badge>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Input Folder:</span>
              <span className="text-sm font-mono truncate max-w-48">
                {mockProject.inputFolder}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Output Folder:</span>
              <span className="text-sm font-mono truncate max-w-48">
                {mockProject.outputFolder}
              </span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Quick Actions</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <Button variant="outline" className="w-full justify-start">
              <EyeIcon className="w-4 h-4 mr-2" />
              Review Rejected Images
            </Button>
            <Button variant="outline" className="w-full justify-start">
              <DownloadIcon className="w-4 h-4 mr-2" />
              Export Results
            </Button>
            <Button variant="outline" className="w-full justify-start">
              <SettingsIcon className="w-4 h-4 mr-2" />
              Adjust Settings
            </Button>
          </CardContent>
        </Card>
      </motion.div>

      {/* Project Details Tabs */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
      >
        <Tabs defaultValue="log" className="space-y-6">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="log">Processing Log</TabsTrigger>
            <TabsTrigger value="history">Session History</TabsTrigger>
          </TabsList>

          <TabsContent value="log">
            <Card>
              <CardHeader>
                <CardTitle>Processing Log</CardTitle>
                <CardDescription>
                  Real-time updates from the processing engine
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="bg-muted/50 rounded-lg p-4 font-mono text-sm space-y-1 max-h-64 overflow-y-auto">
                  <div className="text-muted-foreground">
                    [14:23:45] Starting batch processing...
                  </div>
                  <div className="text-blue-600">
                    [14:23:46] Loading AI models...
                  </div>
                  <div className="text-green-600">
                    [14:23:48] Processing IMG_5432.jpg - Quality: 0.85, Defects: None
                  </div>
                  <div className="text-green-600">
                    [14:23:49] Processing IMG_5433.jpg - Quality: 0.92, Defects: None
                  </div>
                  <div className="text-yellow-600">
                    [14:23:50] Processing IMG_5434.jpg - Quality: 0.65, Defects: Low sharpness
                  </div>
                  <div className="text-muted-foreground">
                    [14:23:51] Batch 1/25 completed. Memory usage: 2.1GB
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="history">
            <SingleProjectHistory projectId={projectId} />
          </TabsContent>
        </Tabs>
      </motion.div>
    </div>
  );
}