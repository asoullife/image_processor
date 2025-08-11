"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Settings, Zap, Balance, Brain, Info } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

interface SimilarityConfigProps {
  sessionId: string;
  onClose: () => void;
}

interface SimilarityConfig {
  use_case: string;
  clip_threshold: number;
  visual_threshold: number;
  identical_threshold: number;
  near_duplicate_threshold: number;
  similar_threshold: number;
  clustering_eps: number;
  min_cluster_size: number;
}

interface UseCase {
  description: string;
  description_en: string;
  clip_threshold: number;
  visual_threshold: number;
  identical_threshold: number;
  near_duplicate_threshold: number;
  similar_threshold: number;
  clustering_eps: number;
}

export function SimilarityConfig({ sessionId, onClose }: SimilarityConfigProps) {
  const [activeTab, setActiveTab] = useState("presets");
  const [customConfig, setCustomConfig] = useState<SimilarityConfig | null>(null);
  const queryClient = useQueryClient();

  // Fetch current configuration
  const { data: configData, isLoading } = useQuery({
    queryKey: ["similarity-config", sessionId],
    queryFn: async () => {
      const response = await fetch(`/api/review/sessions/${sessionId}/similarity-config`);
      return response.json();
    },
  });

  // Update configuration mutation
  const updateConfigMutation = useMutation({
    mutationFn: async (config: Partial<SimilarityConfig>) => {
      const response = await fetch(`/api/review/sessions/${sessionId}/similarity-config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      return response.json();
    },
    onSuccess: (data) => {
      toast.success(data.thai_message || "Configuration updated successfully");
      queryClient.invalidateQueries({ queryKey: ["similarity-config", sessionId] });
      onClose();
    },
    onError: () => {
      toast.error("Failed to update configuration");
    },
  });

  const currentConfig: SimilarityConfig = configData?.current_config || {};
  const useCasePresets: Record<string, UseCase> = configData?.use_case_presets || {};
  const thresholdDescriptions: Record<string, string> = configData?.threshold_descriptions || {};

  const handlePresetSelect = (useCase: string) => {
    updateConfigMutation.mutate({ use_case: useCase });
  };

  const handleCustomConfigUpdate = (key: keyof SimilarityConfig, value: number | string) => {
    setCustomConfig(prev => ({
      ...currentConfig,
      ...prev,
      [key]: value
    }));
  };

  const handleCustomConfigSave = () => {
    if (customConfig) {
      updateConfigMutation.mutate(customConfig);
    }
  };

  const getUseCaseIcon = (useCase: string) => {
    switch (useCase) {
      case "strict": return <Zap className="w-5 h-5 text-red-500" />;
      case "balanced": return <Balance className="w-5 h-5 text-blue-500" />;
      case "lenient": return <Brain className="w-5 h-5 text-green-500" />;
      default: return <Settings className="w-5 h-5" />;
    }
  };

  const getUseCaseColor = (useCase: string) => {
    switch (useCase) {
      case "strict": return "border-red-200 bg-red-50";
      case "balanced": return "border-blue-200 bg-blue-50";
      case "lenient": return "border-green-200 bg-green-50";
      default: return "border-gray-200 bg-gray-50";
    }
  };

  if (isLoading) {
    return (
      <Dialog open onOpenChange={onClose}>
        <DialogContent className="max-w-4xl">
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
            <p className="ml-4">Loading configuration...</p>
          </div>
        </DialogContent>
      </Dialog>
    );
  }

  return (
    <Dialog open onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-2">
            <Settings className="w-5 h-5" />
            <span>Similarity Detection Configuration - ตั้งค่าการตรวจจับความคล้าย</span>
          </DialogTitle>
        </DialogHeader>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="presets">Use Case Presets</TabsTrigger>
            <TabsTrigger value="custom">Custom Settings</TabsTrigger>
          </TabsList>

          <TabsContent value="presets" className="space-y-4">
            <div className="grid gap-4">
              {Object.entries(useCasePresets).map(([key, preset]) => (
                <Card 
                  key={key}
                  className={cn(
                    "cursor-pointer transition-all hover:shadow-md",
                    getUseCaseColor(key),
                    currentConfig.use_case === key && "ring-2 ring-primary"
                  )}
                  onClick={() => handlePresetSelect(key)}
                >
                  <CardHeader className="pb-3">
                    <CardTitle className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        {getUseCaseIcon(key)}
                        <div>
                          <h3 className="font-semibold capitalize">{key}</h3>
                          <p className="text-sm text-muted-foreground font-normal">
                            {preset.description}
                          </p>
                        </div>
                      </div>
                      {currentConfig.use_case === key && (
                        <Badge variant="default">Current</Badge>
                      )}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="pt-0">
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
                      <div>
                        <span className="text-muted-foreground">CLIP:</span>
                        <span className="ml-2 font-medium">{(preset.clip_threshold * 100).toFixed(0)}%</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Visual:</span>
                        <span className="ml-2 font-medium">{(preset.visual_threshold * 100).toFixed(0)}%</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Identical:</span>
                        <span className="ml-2 font-medium">{(preset.identical_threshold * 100).toFixed(0)}%</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Near Dup:</span>
                        <span className="ml-2 font-medium">{(preset.near_duplicate_threshold * 100).toFixed(0)}%</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Similar:</span>
                        <span className="ml-2 font-medium">{(preset.similar_threshold * 100).toFixed(0)}%</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Clustering:</span>
                        <span className="ml-2 font-medium">{preset.clustering_eps.toFixed(1)}</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="custom" className="space-y-6">
            <div className="space-y-6">
              {/* CLIP Threshold */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">CLIP Similarity Threshold</label>
                  <span className="text-sm text-muted-foreground">
                    {((customConfig?.clip_threshold ?? currentConfig.clip_threshold) * 100).toFixed(0)}%
                  </span>
                </div>
                <Slider
                  value={[(customConfig?.clip_threshold ?? currentConfig.clip_threshold) * 100]}
                  onValueChange={([value]) => handleCustomConfigUpdate('clip_threshold', value / 100)}
                  max={100}
                  min={50}
                  step={1}
                  className="w-full"
                />
                <p className="text-xs text-muted-foreground">
                  {thresholdDescriptions.clip_threshold}
                </p>
              </div>

              {/* Visual Threshold */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">Visual Similarity Threshold</label>
                  <span className="text-sm text-muted-foreground">
                    {((customConfig?.visual_threshold ?? currentConfig.visual_threshold) * 100).toFixed(0)}%
                  </span>
                </div>
                <Slider
                  value={[(customConfig?.visual_threshold ?? currentConfig.visual_threshold) * 100]}
                  onValueChange={([value]) => handleCustomConfigUpdate('visual_threshold', value / 100)}
                  max={100}
                  min={50}
                  step={1}
                  className="w-full"
                />
                <p className="text-xs text-muted-foreground">
                  {thresholdDescriptions.visual_threshold}
                </p>
              </div>

              {/* Identical Threshold */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">Identical Images Threshold</label>
                  <span className="text-sm text-muted-foreground">
                    {((customConfig?.identical_threshold ?? currentConfig.identical_threshold) * 100).toFixed(0)}%
                  </span>
                </div>
                <Slider
                  value={[(customConfig?.identical_threshold ?? currentConfig.identical_threshold) * 100]}
                  onValueChange={([value]) => handleCustomConfigUpdate('identical_threshold', value / 100)}
                  max={100}
                  min={80}
                  step={1}
                  className="w-full"
                />
                <p className="text-xs text-muted-foreground">
                  {thresholdDescriptions.identical_threshold}
                </p>
              </div>

              {/* Near Duplicate Threshold */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">Near Duplicate Threshold</label>
                  <span className="text-sm text-muted-foreground">
                    {((customConfig?.near_duplicate_threshold ?? currentConfig.near_duplicate_threshold) * 100).toFixed(0)}%
                  </span>
                </div>
                <Slider
                  value={[(customConfig?.near_duplicate_threshold ?? currentConfig.near_duplicate_threshold) * 100]}
                  onValueChange={([value]) => handleCustomConfigUpdate('near_duplicate_threshold', value / 100)}
                  max={100}
                  min={70}
                  step={1}
                  className="w-full"
                />
                <p className="text-xs text-muted-foreground">
                  {thresholdDescriptions.near_duplicate_threshold}
                </p>
              </div>

              {/* Similar Threshold */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">Similar Images Threshold</label>
                  <span className="text-sm text-muted-foreground">
                    {((customConfig?.similar_threshold ?? currentConfig.similar_threshold) * 100).toFixed(0)}%
                  </span>
                </div>
                <Slider
                  value={[(customConfig?.similar_threshold ?? currentConfig.similar_threshold) * 100]}
                  onValueChange={([value]) => handleCustomConfigUpdate('similar_threshold', value / 100)}
                  max={100}
                  min={50}
                  step={1}
                  className="w-full"
                />
                <p className="text-xs text-muted-foreground">
                  {thresholdDescriptions.similar_threshold}
                </p>
              </div>

              {/* Clustering EPS */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">Clustering Distance</label>
                  <span className="text-sm text-muted-foreground">
                    {(customConfig?.clustering_eps ?? currentConfig.clustering_eps).toFixed(2)}
                  </span>
                </div>
                <Slider
                  value={[(customConfig?.clustering_eps ?? currentConfig.clustering_eps) * 100]}
                  onValueChange={([value]) => handleCustomConfigUpdate('clustering_eps', value / 100)}
                  max={50}
                  min={10}
                  step={1}
                  className="w-full"
                />
                <p className="text-xs text-muted-foreground">
                  {thresholdDescriptions.clustering_eps}
                </p>
              </div>

              {/* Min Cluster Size */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">Minimum Cluster Size</label>
                  <span className="text-sm text-muted-foreground">
                    {customConfig?.min_cluster_size ?? currentConfig.min_cluster_size}
                  </span>
                </div>
                <Slider
                  value={[customConfig?.min_cluster_size ?? currentConfig.min_cluster_size]}
                  onValueChange={([value]) => handleCustomConfigUpdate('min_cluster_size', value)}
                  max={10}
                  min={2}
                  step={1}
                  className="w-full"
                />
                <p className="text-xs text-muted-foreground">
                  {thresholdDescriptions.min_cluster_size}
                </p>
              </div>
            </div>

            <div className="flex justify-end space-x-2 pt-4 border-t">
              <Button variant="outline" onClick={onClose}>
                Cancel
              </Button>
              <Button 
                onClick={handleCustomConfigSave}
                disabled={updateConfigMutation.isPending}
              >
                {updateConfigMutation.isPending ? "Saving..." : "Save Custom Settings"}
              </Button>
            </div>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}