"use client";

import { motion } from "framer-motion";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  PlusIcon, 
  FolderIcon, 
  ClockIcon, 
  CheckCircleIcon,
  XCircleIcon,
  PlayIcon,
  PauseIcon,
  ActivityIcon
} from "lucide-react";
import Link from "next/link";
import { MultiSessionDashboard } from "@/components/dashboard/MultiSessionDashboard";
import { ProjectHistory } from "@/components/projects/ProjectHistory";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

// Mock data - will be replaced with real data from API
const mockProjects = [
  {
    id: "1",
    name: "Wedding Photos Batch 1",
    status: "completed",
    totalImages: 1250,
    processedImages: 1250,
    approvedImages: 1180,
    rejectedImages: 70,
    createdAt: "2024-01-15T10:30:00Z",
    completedAt: "2024-01-15T12:45:00Z",
    inputFolder: "/Users/photographer/wedding-batch-1",
    outputFolder: "/Users/photographer/wedding-batch-1_processed"
  },
  {
    id: "2", 
    name: "Nature Photography Collection",
    status: "processing",
    totalImages: 3200,
    processedImages: 1850,
    approvedImages: 1650,
    rejectedImages: 200,
    createdAt: "2024-01-16T09:00:00Z",
    inputFolder: "/Users/photographer/nature-collection",
    outputFolder: "/Users/photographer/nature-collection_processed"
  },
  {
    id: "3",
    name: "Portrait Session Q1",
    status: "paused",
    totalImages: 800,
    processedImages: 320,
    approvedImages: 280,
    rejectedImages: 40,
    createdAt: "2024-01-14T14:20:00Z",
    inputFolder: "/Users/photographer/portraits-q1",
    outputFolder: "/Users/photographer/portraits-q1_processed"
  }
];

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

const getStatusIcon = (status: string) => {
  switch (status) {
    case "completed":
      return <CheckCircleIcon className="w-5 h-5 text-green-500" />;
    case "processing":
      return <PlayIcon className="w-5 h-5 text-blue-500" />;
    case "paused":
      return <PauseIcon className="w-5 h-5 text-yellow-500" />;
    case "failed":
      return <XCircleIcon className="w-5 h-5 text-red-500" />;
    default:
      return <ClockIcon className="w-5 h-5 text-gray-500" />;
  }
};

export default function ProjectsPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="flex justify-between items-center mb-8"
      >
        <div>
          <h1 className="text-3xl font-bold mb-2">Projects</h1>
          <p className="text-muted-foreground">
            Manage your image processing projects and view results
          </p>
        </div>
        <Button asChild>
          <Link href="/projects/new">
            <PlusIcon className="w-4 h-4 mr-2" />
            New Project
          </Link>
        </Button>
      </motion.div>

      {/* Main Content Tabs */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
      >
        <Tabs defaultValue="projects" className="space-y-6">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="projects" className="flex items-center gap-2">
              <FolderIcon className="w-4 h-4" />
              All Projects
            </TabsTrigger>
            <TabsTrigger value="active" className="flex items-center gap-2">
              <ActivityIcon className="w-4 h-4" />
              Active Sessions
            </TabsTrigger>
            <TabsTrigger value="history" className="flex items-center gap-2">
              <ClockIcon className="w-4 h-4" />
              History
            </TabsTrigger>
          </TabsList>

          <TabsContent value="projects">
            {/* Projects Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {mockProjects.map((project, index) => (
          <motion.div
            key={project.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 + index * 0.1 }}
          >
            <Card className="hover:shadow-lg transition-all duration-300 cursor-pointer group">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    {getStatusIcon(project.status)}
                    <div>
                      <CardTitle className="text-lg group-hover:text-primary transition-colors">
                        {project.name}
                      </CardTitle>
                      <CardDescription className="flex items-center gap-1 mt-1">
                        <FolderIcon className="w-3 h-3" />
                        {project.totalImages} images
                      </CardDescription>
                    </div>
                  </div>
                  {getStatusBadge(project.status)}
                </div>
              </CardHeader>
              
              <CardContent>
                {/* Progress Stats */}
                <div className="space-y-3 mb-4">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Progress</span>
                    <span className="font-medium">
                      {project.processedImages}/{project.totalImages}
                    </span>
                  </div>
                  
                  <div className="w-full bg-secondary rounded-full h-2">
                    <div 
                      className="bg-primary h-2 rounded-full transition-all duration-300"
                      style={{ 
                        width: `${(project.processedImages / project.totalImages) * 100}%` 
                      }}
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-green-500 rounded-full" />
                      <span className="text-muted-foreground">Approved:</span>
                      <span className="font-medium text-green-600">
                        {project.approvedImages}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-red-500 rounded-full" />
                      <span className="text-muted-foreground">Rejected:</span>
                      <span className="font-medium text-red-600">
                        {project.rejectedImages}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex gap-2">
                  <Button 
                    variant="outline" 
                    size="sm" 
                    className="flex-1"
                    asChild
                  >
                    <Link href={`/projects/${project.id}`}>
                      View Details
                    </Link>
                  </Button>
                  
                  {project.status === "processing" && (
                    <Button variant="outline" size="sm">
                      <PauseIcon className="w-3 h-3" />
                    </Button>
                  )}
                  
                  {project.status === "paused" && (
                    <Button variant="outline" size="sm">
                      <PlayIcon className="w-3 h-3" />
                    </Button>
                  )}
                </div>

                {/* Timestamps */}
                <div className="mt-4 pt-4 border-t text-xs text-muted-foreground">
                  <div className="flex justify-between">
                    <span>Created: {new Date(project.createdAt).toLocaleDateString()}</span>
                    {project.completedAt && (
                      <span>Completed: {new Date(project.completedAt).toLocaleDateString()}</span>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
            </div>

            {/* Empty State */}
            {mockProjects.length === 0 && (
              <div className="text-center py-12">
                <FolderIcon className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-semibold mb-2">No projects yet</h3>
                <p className="text-muted-foreground mb-6">
                  Create your first image processing project to get started
                </p>
                <Button asChild>
                  <Link href="/projects/new">
                    <PlusIcon className="w-4 h-4 mr-2" />
                    Create Project
                  </Link>
                </Button>
              </div>
            )}
          </TabsContent>

          <TabsContent value="active">
            <MultiSessionDashboard />
          </TabsContent>

          <TabsContent value="history">
            <ProjectHistory showProjectNames={true} limit={100} />
          </TabsContent>
        </Tabs>
      </motion.div>


    </div>
  );
}