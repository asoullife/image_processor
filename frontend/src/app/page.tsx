"use client";

import { motion } from "framer-motion";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  ImageIcon, 
  BrainIcon, 
  ShieldCheckIcon, 
  TrendingUpIcon,
  PlayIcon,
  PlusIcon,
  HistoryIcon,
  SettingsIcon
} from "lucide-react";
import Link from "next/link";

const features = [
  {
    icon: ImageIcon,
    title: "AI-Enhanced Quality Analysis",
    description: "Advanced image quality detection using TensorFlow models with GPU acceleration",
    color: "text-blue-500"
  },
  {
    icon: BrainIcon,
    title: "Smart Defect Detection",
    description: "YOLO v8 powered object detection and anomaly identification",
    color: "text-purple-500"
  },
  {
    icon: ShieldCheckIcon,
    title: "File Integrity Protection",
    description: "100% safe processing - original files are never modified",
    color: "text-green-500"
  },
  {
    icon: TrendingUpIcon,
    title: "Real-time Monitoring",
    description: "Live progress updates and performance metrics via WebSocket",
    color: "text-orange-500"
  }
];

const quickActions = [
  {
    icon: PlusIcon,
    title: "New Project",
    description: "Start processing a new image collection",
    href: "/projects/new",
    variant: "default" as const
  },
  {
    icon: HistoryIcon,
    title: "Recent Projects",
    description: "View and manage your processing history",
    href: "/projects",
    variant: "outline" as const
  },
  {
    icon: SettingsIcon,
    title: "Settings",
    description: "Configure processing preferences",
    href: "/settings",
    variant: "outline" as const
  }
];

export default function HomePage() {
  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center mb-12"
      >
        <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
          Adobe Stock Image Processor
        </h1>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
          AI-powered desktop application for processing and analyzing large image collections 
          with advanced quality detection and compliance checking
        </p>
        <div className="flex justify-center gap-2 mt-4">
          <Badge variant="secondary">AI Enhanced</Badge>
          <Badge variant="secondary">GPU Accelerated</Badge>
          <Badge variant="secondary">Real-time Monitoring</Badge>
        </div>
      </motion.div>

      {/* Quick Actions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12"
      >
        {quickActions.map((action, index) => (
          <Link key={action.title} href={action.href}>
            <Card className="hover:shadow-lg transition-all duration-300 cursor-pointer group">
              <CardHeader className="text-center">
                <div className="mx-auto w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
                  <action.icon className="w-6 h-6 text-primary" />
                </div>
                <CardTitle className="text-lg">{action.title}</CardTitle>
                <CardDescription>{action.description}</CardDescription>
              </CardHeader>
              <CardContent className="pt-0">
                <Button 
                  variant={action.variant} 
                  className="w-full"
                  asChild
                >
                  <span>
                    {action.title === "New Project" && <PlayIcon className="w-4 h-4 mr-2" />}
                    Get Started
                  </span>
                </Button>
              </CardContent>
            </Card>
          </Link>
        ))}
      </motion.div>

      {/* Features Grid */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="mb-12"
      >
        <h2 className="text-2xl font-semibold text-center mb-8">Key Features</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.3 + index * 0.1 }}
            >
              <Card className="h-full hover:shadow-md transition-shadow">
                <CardHeader className="text-center">
                  <div className="mx-auto w-12 h-12 bg-background rounded-lg flex items-center justify-center mb-4">
                    <feature.icon className={`w-6 h-6 ${feature.color}`} />
                  </div>
                  <CardTitle className="text-lg">{feature.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-center">
                    {feature.description}
                  </CardDescription>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* System Status */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
              System Status
            </CardTitle>
            <CardDescription>
              Backend services and AI models status
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                <span className="text-sm font-medium">Backend API</span>
                <Badge variant="success">Online</Badge>
              </div>
              <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                <span className="text-sm font-medium">Database</span>
                <Badge variant="success">Connected</Badge>
              </div>
              <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                <span className="text-sm font-medium">AI Models</span>
                <Badge variant="warning">Loading</Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}