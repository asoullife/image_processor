"use client";

import { motion } from "framer-motion";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { 
  FolderIcon, 
  SettingsIcon, 
  PlayIcon,
  ArrowLeftIcon,
  CheckIcon
} from "lucide-react";
import Link from "next/link";
import { useState } from "react";

const performanceModes = [
  {
    id: "speed",
    name: "Speed Mode",
    description: "Fast processing with basic quality checks",
    batchSize: 50,
    features: ["Basic quality analysis", "Fast defect detection", "Simple similarity check"],
    recommended: false
  },
  {
    id: "balanced", 
    name: "Balanced Mode",
    description: "Optimal balance of speed and accuracy",
    batchSize: 20,
    features: ["AI-enhanced quality analysis", "Advanced defect detection", "Perceptual similarity matching"],
    recommended: true
  },
  {
    id: "smart",
    name: "Smart Mode", 
    description: "Maximum accuracy with all AI features",
    batchSize: "Auto",
    features: ["Full AI analysis", "Deep learning similarity", "Comprehensive compliance check"],
    recommended: false
  }
];

export default function NewProjectPage() {
  const [currentStep, setCurrentStep] = useState(1);
  const [formData, setFormData] = useState({
    name: "",
    description: "",
    inputFolder: "",
    outputFolder: "",
    performanceMode: "balanced"
  });

  // Auto-generate output folder when input folder changes
  const handleInputFolderChange = (value: string) => {
    setFormData(prev => {
      const outputFolder = value ? `${value}_processed` : "";
      return {
        ...prev,
        inputFolder: value,
        outputFolder: outputFolder
      };
    });
  };

  const handleNext = () => {
    if (currentStep < 4) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handleBack = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSubmit = () => {
    // TODO: Submit to API
    console.log("Creating project:", formData);
  };

  const steps = [
    { number: 1, title: "Project Setup", description: "Basic project information" },
    { number: 2, title: "Folder Selection", description: "Choose input and output folders" },
    { number: 3, title: "AI Settings", description: "Configure processing options" },
    { number: 4, title: "Review & Start", description: "Confirm and start processing" }
  ];

  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="flex items-center gap-4 mb-8"
      >
        <Button variant="outline" size="sm" asChild>
          <Link href="/projects">
            <ArrowLeftIcon className="w-4 h-4 mr-2" />
            Back to Projects
          </Link>
        </Button>
        <div>
          <h1 className="text-3xl font-bold">Create New Project</h1>
          <p className="text-muted-foreground">
            Set up a new image processing project with AI-enhanced analysis
          </p>
        </div>
      </motion.div>

      {/* Progress Steps */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="mb-8"
      >
        <div className="flex items-center justify-between">
          {steps.map((step, index) => (
            <div key={step.number} className="flex items-center">
              <div className={`flex items-center justify-center w-10 h-10 rounded-full border-2 transition-colors ${
                currentStep >= step.number 
                  ? "bg-primary border-primary text-primary-foreground" 
                  : "border-muted-foreground text-muted-foreground"
              }`}>
                {currentStep > step.number ? (
                  <CheckIcon className="w-5 h-5" />
                ) : (
                  <span className="text-sm font-medium">{step.number}</span>
                )}
              </div>
              <div className="ml-3">
                <p className={`text-sm font-medium ${
                  currentStep >= step.number ? "text-foreground" : "text-muted-foreground"
                }`}>
                  {step.title}
                </p>
                <p className="text-xs text-muted-foreground">{step.description}</p>
              </div>
              {index < steps.length - 1 && (
                <div className={`w-16 h-0.5 mx-4 ${
                  currentStep > step.number ? "bg-primary" : "bg-muted"
                }`} />
              )}
            </div>
          ))}
        </div>
      </motion.div>

      {/* Step Content */}
      <motion.div
        key={currentStep}
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.3 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {currentStep === 1 && <SettingsIcon className="w-5 h-5" />}
              {currentStep === 2 && <FolderIcon className="w-5 h-5" />}
              {currentStep === 3 && <SettingsIcon className="w-5 h-5" />}
              {currentStep === 4 && <PlayIcon className="w-5 h-5" />}
              {steps[currentStep - 1].title}
            </CardTitle>
            <CardDescription>
              {steps[currentStep - 1].description}
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-6">
            {/* Step 1: Project Setup */}
            {currentStep === 1 && (
              <div className="space-y-4">
                <div>
                  <Label htmlFor="projectName">Project Name</Label>
                  <Input
                    id="projectName"
                    placeholder="e.g., Wedding Photos Batch 1"
                    value={formData.name}
                    onChange={(e) => setFormData({...formData, name: e.target.value})}
                  />
                </div>
                <div>
                  <Label htmlFor="projectDescription">Description (Optional)</Label>
                  <Input
                    id="projectDescription"
                    placeholder="Brief description of this image collection"
                    value={formData.description}
                    onChange={(e) => setFormData({...formData, description: e.target.value})}
                  />
                </div>
              </div>
            )}

            {/* Step 2: Folder Selection */}
            {currentStep === 2 && (
              <div className="space-y-4">
                <div>
                  <Label htmlFor="inputFolder">Input Folder</Label>
                  <div className="flex gap-2">
                    <Input
                      id="inputFolder"
                      placeholder="/path/to/your/images"
                      value={formData.inputFolder}
                      onChange={(e) => handleInputFolderChange(e.target.value)}
                    />
                    <Button variant="outline">Browse</Button>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    Select the folder containing your images to process
                  </p>
                </div>
                <div>
                  <Label htmlFor="outputFolder">Output Folder</Label>
                  <div className="flex gap-2">
                    <Input
                      id="outputFolder"
                      placeholder="/path/to/output/folder"
                      value={formData.outputFolder}
                      onChange={(e) => setFormData({...formData, outputFolder: e.target.value})}
                    />
                    <Button variant="outline">Browse</Button>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    Approved images will be copied here with "_processed" suffix
                  </p>
                </div>
              </div>
            )}

            {/* Step 3: AI Settings */}
            {currentStep === 3 && (
              <div className="space-y-4">
                <div>
                  <Label>Performance Mode</Label>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-2">
                    {performanceModes.map((mode) => (
                      <div
                        key={mode.id}
                        className={`relative p-4 border rounded-lg cursor-pointer transition-all ${
                          formData.performanceMode === mode.id
                            ? "border-primary bg-primary/5"
                            : "border-muted hover:border-muted-foreground"
                        }`}
                        onClick={() => setFormData({...formData, performanceMode: mode.id})}
                      >
                        {mode.recommended && (
                          <Badge className="absolute -top-2 -right-2" variant="default">
                            Recommended
                          </Badge>
                        )}
                        <div className="space-y-2">
                          <h4 className="font-medium">{mode.name}</h4>
                          <p className="text-sm text-muted-foreground">{mode.description}</p>
                          <div className="text-xs text-muted-foreground">
                            Batch size: {mode.batchSize}
                          </div>
                          <ul className="text-xs space-y-1">
                            {mode.features.map((feature, index) => (
                              <li key={index} className="flex items-center gap-1">
                                <CheckIcon className="w-3 h-3 text-green-500" />
                                {feature}
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Step 4: Review & Start */}
            {currentStep === 4 && (
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-medium mb-2">Project Details</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Name:</span>
                        <span>{formData.name || "Untitled Project"}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Description:</span>
                        <span>{formData.description || "No description"}</span>
                      </div>
                    </div>
                  </div>
                  <div>
                    <h4 className="font-medium mb-2">Processing Settings</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Mode:</span>
                        <span>{performanceModes.find(m => m.id === formData.performanceMode)?.name}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Input:</span>
                        <span className="truncate max-w-32">{formData.inputFolder || "Not selected"}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Output:</span>
                        <span className="truncate max-w-32">{formData.outputFolder || "Not selected"}</span>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="p-4 bg-muted/50 rounded-lg">
                  <h4 className="font-medium mb-2">What happens next?</h4>
                  <ul className="text-sm space-y-1 text-muted-foreground">
                    <li>• Images will be scanned and analyzed using AI models</li>
                    <li>• You can monitor progress in real-time via the web interface</li>
                    <li>• Approved images will be copied to the output folder</li>
                    <li>• You can review and override AI decisions at any time</li>
                  </ul>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Navigation Buttons */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="flex justify-between mt-8"
      >
        <Button
          variant="outline"
          onClick={handleBack}
          disabled={currentStep === 1}
        >
          Previous
        </Button>
        
        {currentStep < 4 ? (
          <Button onClick={handleNext}>
            Next
          </Button>
        ) : (
          <Button onClick={handleSubmit}>
            <PlayIcon className="w-4 h-4 mr-2" />
            Start Processing
          </Button>
        )}
      </motion.div>
    </div>
  );
}