'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { toast } from 'sonner';
import { Loader2, Cpu, HardDrive, MemoryStick, Zap, AlertTriangle, CheckCircle, Download, Upload, RotateCcw, Settings2 } from 'lucide-react';
import { api } from '@/lib/api';

interface SystemHealth {
  status: 'healthy' | 'warning' | 'critical' | 'error';
  cpu_usage_percent: number;
  memory_usage_percent: number;
  disk_usage_percent: number;
  memory_available_gb: number;
  disk_free_gb: number;
  gpu_metrics?: Array<{
    name: string;
    load: number;
    memory_used_percent: number;
    temperature: number;
  }>;
}

interface HardwareInfo {
  platform: string;
  processor: string;
  cpu_count_physical: number;
  cpu_count_logical: number;
  memory_total_gb: number;
  gpus: Array<{
    name: string;
    memory_total: number;
  }>;
}

interface PerformanceRecommendations {
  recommended_mode: string;
  recommended_batch_size: number;
  recommended_workers: number;
  gpu_acceleration: boolean;
  memory_limit_gb: number;
  warnings: string[];
  optimizations: string[];
}

export default function SettingsPage() {
  const [settings, setSettings] = useState({
    // Processing settings
    performance_mode: 'balanced',
    batch_size: 20,
    max_workers: 4,
    gpu_enabled: true,
    memory_limit_gb: 8.0,
    quality_threshold: 0.7,
    similarity_threshold: 0.85,
    defect_confidence_threshold: 0.8,
    use_ai_enhancement: true,
    fallback_to_opencv: true,
    model_precision: 'fp16',
    
    // System settings
    log_level: 'INFO',
    auto_cleanup: true,
    backup_checkpoints: true,
    max_file_size_mb: 100,
    
    // UI settings
    theme: 'light',
    language: 'th',
    auto_refresh_interval: 5000,
    thumbnail_size: 200,
    items_per_page: 50,
    show_confidence_scores: true
  });

  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [hardwareInfo, setHardwareInfo] = useState<HardwareInfo | null>(null);
  const [recommendations, setRecommendations] = useState<PerformanceRecommendations | null>(null);
  const [loading, setLoading] = useState(false);
  const [optimizing, setOptimizing] = useState(false);

  useEffect(() => {
    loadSettings();
    loadSystemHealth();
    loadHardwareInfo();
    loadRecommendations();
  }, []);

  const loadSettings = async () => {
    try {
      const response = await api.get('/settings/');
      if (response.data.success) {
        const data = response.data.data;
        setSettings({
          ...data.processing,
          ...data.system,
          ...data.ui
        });
      }
    } catch (error) {
      console.error('Failed to load settings:', error);
      toast.error('Failed to load settings');
    }
  };

  const loadSystemHealth = async () => {
    try {
      const response = await api.get('/settings/health');
      if (response.data.success) {
        setSystemHealth(response.data.data);
      }
    } catch (error) {
      console.error('Failed to load system health:', error);
    }
  };

  const loadHardwareInfo = async () => {
    try {
      const response = await api.get('/settings/hardware');
      if (response.data.success) {
        setHardwareInfo(response.data.data);
      }
    } catch (error) {
      console.error('Failed to load hardware info:', error);
    }
  };

  const loadRecommendations = async () => {
    try {
      const response = await api.get('/settings/recommendations');
      if (response.data.success) {
        setRecommendations(response.data.data.recommendations);
      }
    } catch (error) {
      console.error('Failed to load recommendations:', error);
    }
  };

  const handleSave = async () => {
    setLoading(true);
    try {
      // Update processing settings
      await api.put('/settings/processing', {
        performance_mode: settings.performance_mode,
        batch_size: settings.batch_size,
        max_workers: settings.max_workers,
        gpu_enabled: settings.gpu_enabled,
        memory_limit_gb: settings.memory_limit_gb,
        quality_threshold: settings.quality_threshold,
        similarity_threshold: settings.similarity_threshold,
        defect_confidence_threshold: settings.defect_confidence_threshold,
        use_ai_enhancement: settings.use_ai_enhancement,
        fallback_to_opencv: settings.fallback_to_opencv,
        model_precision: settings.model_precision
      });

      // Update system settings
      await api.put('/settings/system', {
        log_level: settings.log_level,
        auto_cleanup: settings.auto_cleanup,
        backup_checkpoints: settings.backup_checkpoints,
        max_file_size_mb: settings.max_file_size_mb
      });

      // Update UI settings
      await api.put('/settings/ui', {
        theme: settings.theme,
        language: settings.language,
        auto_refresh_interval: settings.auto_refresh_interval,
        thumbnail_size: settings.thumbnail_size,
        items_per_page: settings.items_per_page,
        show_confidence_scores: settings.show_confidence_scores
      });

      toast.success('Settings saved successfully');
    } catch (error) {
      console.error('Failed to save settings:', error);
      toast.error('Failed to save settings');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    setLoading(true);
    try {
      await api.post('/settings/reset');
      await loadSettings();
      toast.success('Settings reset to defaults');
    } catch (error) {
      console.error('Failed to reset settings:', error);
      toast.error('Failed to reset settings');
    } finally {
      setLoading(false);
    }
  };

  const handleOptimize = async () => {
    setOptimizing(true);
    try {
      const response = await api.post('/settings/optimize');
      if (response.data.success) {
        await loadSettings();
        await loadRecommendations();
        toast.success('Settings optimized for your hardware');
      }
    } catch (error) {
      console.error('Failed to optimize settings:', error);
      toast.error('Failed to optimize settings');
    } finally {
      setOptimizing(false);
    }
  };

  const handlePerformanceModeChange = async (mode: string) => {
    try {
      const response = await api.post('/settings/performance-mode', { mode });
      if (response.data.success) {
        setSettings(prev => ({
          ...prev,
          ...response.data.data.settings
        }));
        toast.success(`Performance mode set to ${mode}`);
      }
    } catch (error) {
      console.error('Failed to set performance mode:', error);
      toast.error('Failed to set performance mode');
    }
  };

  const exportSettings = async () => {
    try {
      const response = await api.get('/settings/export');
      if (response.data.success) {
        const blob = new Blob([JSON.stringify(response.data.data, null, 2)], {
          type: 'application/json'
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = response.data.filename || 'settings.json';
        a.click();
        URL.revokeObjectURL(url);
        toast.success('Settings exported successfully');
      }
    } catch (error) {
      console.error('Failed to export settings:', error);
      toast.error('Failed to export settings');
    }
  };

  const getHealthStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-600';
      case 'warning': return 'text-yellow-600';
      case 'critical': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getHealthStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'warning': return <AlertTriangle className="h-4 w-4 text-yellow-600" />;
      case 'critical': return <AlertTriangle className="h-4 w-4 text-red-600" />;
      default: return <AlertTriangle className="h-4 w-4 text-gray-600" />;
    }
  };

  return (
    <div className="container mx-auto py-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold">Settings</h1>
          <p className="text-muted-foreground">Configure your processing preferences and system settings</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={handleReset} disabled={loading}>
            <RotateCcw className="h-4 w-4 mr-2" />
            Reset to Defaults
          </Button>
          <Button onClick={handleSave} disabled={loading}>
            {loading ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Settings2 className="h-4 w-4 mr-2" />}
            Save Changes
          </Button>
        </div>
      </div>

      <Tabs defaultValue="processing" className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="processing">Processing</TabsTrigger>
          <TabsTrigger value="system">System</TabsTrigger>
          <TabsTrigger value="ui">Interface</TabsTrigger>
          <TabsTrigger value="health">Health</TabsTrigger>
          <TabsTrigger value="advanced">Advanced</TabsTrigger>
        </TabsList>

        <TabsContent value="processing" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Performance Settings</CardTitle>
              <CardDescription>
                Configure processing performance and resource usage
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-2 gap-6">
                <div className="space-y-2">
                  <Label htmlFor="performance-mode">Performance Mode</Label>
                  <Select 
                    value={settings.performance_mode} 
                    onValueChange={handlePerformanceModeChange}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="speed">
                        <div className="flex items-center gap-2">
                          <Zap className="h-4 w-4" />
                          Speed Mode
                          <Badge variant="secondary">Fast</Badge>
                        </div>
                      </SelectItem>
                      <SelectItem value="balanced">
                        <div className="flex items-center gap-2">
                          <Settings2 className="h-4 w-4" />
                          Balanced Mode
                          <Badge variant="default">Recommended</Badge>
                        </div>
                      </SelectItem>
                      <SelectItem value="smart">
                        <div className="flex items-center gap-2">
                          <Cpu className="h-4 w-4" />
                          Smart Mode
                          <Badge variant="outline">Thorough</Badge>
                        </div>
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="batch-size">Batch Size</Label>
                  <Input
                    id="batch-size"
                    type="number"
                    value={settings.batch_size}
                    onChange={(e) => setSettings(prev => ({ 
                      ...prev, 
                      batch_size: parseInt(e.target.value) 
                    }))}
                    min="1"
                    max="1000"
                  />
                  <p className="text-xs text-muted-foreground">
                    Number of images processed simultaneously
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="max-workers">Max Workers</Label>
                  <Input
                    id="max-workers"
                    type="number"
                    value={settings.max_workers}
                    onChange={(e) => setSettings(prev => ({ 
                      ...prev, 
                      max_workers: parseInt(e.target.value) 
                    }))}
                    min="1"
                    max="32"
                  />
                  <p className="text-xs text-muted-foreground">
                    Number of parallel processing threads
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="memory-limit">Memory Limit (GB)</Label>
                  <Input
                    id="memory-limit"
                    type="number"
                    step="0.1"
                    value={settings.memory_limit_gb}
                    onChange={(e) => setSettings(prev => ({ 
                      ...prev, 
                      memory_limit_gb: parseFloat(e.target.value) 
                    }))}
                    min="1"
                    max="64"
                  />
                  <p className="text-xs text-muted-foreground">
                    Maximum memory usage for processing
                  </p>
                </div>
              </div>

              <Separator />

              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>GPU Acceleration</Label>
                    <p className="text-sm text-muted-foreground">
                      Use GPU for faster AI processing
                    </p>
                  </div>
                  <Switch
                    checked={settings.gpu_enabled}
                    onCheckedChange={(checked) => 
                      setSettings(prev => ({ ...prev, gpu_enabled: checked }))
                    }
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>AI Enhancement</Label>
                    <p className="text-sm text-muted-foreground">
                      Use advanced AI models for better accuracy
                    </p>
                  </div>
                  <Switch
                    checked={settings.use_ai_enhancement}
                    onCheckedChange={(checked) => 
                      setSettings(prev => ({ ...prev, use_ai_enhancement: checked }))
                    }
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>OpenCV Fallback</Label>
                    <p className="text-sm text-muted-foreground">
                      Fallback to OpenCV if AI models fail
                    </p>
                  </div>
                  <Switch
                    checked={settings.fallback_to_opencv}
                    onCheckedChange={(checked) => 
                      setSettings(prev => ({ ...prev, fallback_to_opencv: checked }))
                    }
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Quality Thresholds</CardTitle>
              <CardDescription>
                Configure quality and similarity detection thresholds
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-2 gap-6">
                <div className="space-y-2">
                  <Label htmlFor="quality-threshold">Quality Threshold</Label>
                  <Input
                    id="quality-threshold"
                    type="number"
                    step="0.01"
                    value={settings.quality_threshold}
                    onChange={(e) => setSettings(prev => ({ 
                      ...prev, 
                      quality_threshold: parseFloat(e.target.value) 
                    }))}
                    min="0"
                    max="1"
                  />
                  <p className="text-xs text-muted-foreground">
                    Higher values are more strict (0.0 - 1.0)
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="similarity-threshold">Similarity Threshold</Label>
                  <Input
                    id="similarity-threshold"
                    type="number"
                    step="0.01"
                    value={settings.similarity_threshold}
                    onChange={(e) => setSettings(prev => ({ 
                      ...prev, 
                      similarity_threshold: parseFloat(e.target.value) 
                    }))}
                    min="0"
                    max="1"
                  />
                  <p className="text-xs text-muted-foreground">
                    Higher values detect more similar images (0.0 - 1.0)
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="defect-threshold">Defect Confidence Threshold</Label>
                  <Input
                    id="defect-threshold"
                    type="number"
                    step="0.01"
                    value={settings.defect_confidence_threshold}
                    onChange={(e) => setSettings(prev => ({ 
                      ...prev, 
                      defect_confidence_threshold: parseFloat(e.target.value) 
                    }))}
                    min="0"
                    max="1"
                  />
                  <p className="text-xs text-muted-foreground">
                    Minimum confidence for defect detection (0.0 - 1.0)
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="model-precision">Model Precision</Label>
                  <Select 
                    value={settings.model_precision} 
                    onValueChange={(value) => 
                      setSettings(prev => ({ ...prev, model_precision: value }))
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="fp32">FP32 (Highest Quality)</SelectItem>
                      <SelectItem value="fp16">FP16 (Balanced)</SelectItem>
                      <SelectItem value="int8">INT8 (Fastest)</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    Model precision affects speed vs accuracy
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {recommendations && (
            <Card>
              <CardHeader>
                <CardTitle>Hardware Recommendations</CardTitle>
                <CardDescription>
                  Optimized settings based on your hardware
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {recommendations.optimizations.length > 0 && (
                  <div>
                    <h4 className="font-medium text-green-600 mb-2">Optimizations Available:</h4>
                    <ul className="space-y-1">
                      {recommendations.optimizations.map((opt, index) => (
                        <li key={index} className="text-sm flex items-center gap-2">
                          <CheckCircle className="h-3 w-3 text-green-600" />
                          {opt}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {recommendations.warnings.length > 0 && (
                  <div>
                    <h4 className="font-medium text-yellow-600 mb-2">Warnings:</h4>
                    <ul className="space-y-1">
                      {recommendations.warnings.map((warning, index) => (
                        <li key={index} className="text-sm flex items-center gap-2">
                          <AlertTriangle className="h-3 w-3 text-yellow-600" />
                          {warning}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                <Button onClick={handleOptimize} disabled={optimizing} className="w-full">
                  {optimizing ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Zap className="h-4 w-4 mr-2" />
                  )}
                  Auto-Optimize Settings
                </Button>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="system" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>System Configuration</CardTitle>
              <CardDescription>
                Configure system behavior and logging
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-2 gap-6">
                <div className="space-y-2">
                  <Label htmlFor="log-level">Log Level</Label>
                  <Select value={settings.log_level} onValueChange={(value) => 
                    setSettings(prev => ({ ...prev, log_level: value }))
                  }>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="DEBUG">Debug (Verbose)</SelectItem>
                      <SelectItem value="INFO">Info (Normal)</SelectItem>
                      <SelectItem value="WARNING">Warning (Important)</SelectItem>
                      <SelectItem value="ERROR">Error (Critical Only)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="max-file-size">Max File Size (MB)</Label>
                  <Input
                    id="max-file-size"
                    type="number"
                    value={settings.max_file_size_mb}
                    onChange={(e) => setSettings(prev => ({ 
                      ...prev, 
                      max_file_size_mb: parseInt(e.target.value) 
                    }))}
                    min="1"
                    max="1000"
                  />
                  <p className="text-xs text-muted-foreground">
                    Maximum size for individual image files
                  </p>
                </div>
              </div>

              <Separator />

              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Auto Cleanup</Label>
                    <p className="text-sm text-muted-foreground">
                      Automatically clean up temporary files after processing
                    </p>
                  </div>
                  <Switch
                    checked={settings.auto_cleanup}
                    onCheckedChange={(checked) => 
                      setSettings(prev => ({ ...prev, auto_cleanup: checked }))
                    }
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Backup Checkpoints</Label>
                    <p className="text-sm text-muted-foreground">
                      Keep backup copies of processing checkpoints for recovery
                    </p>
                  </div>
                  <Switch
                    checked={settings.backup_checkpoints}
                    onCheckedChange={(checked) => 
                      setSettings(prev => ({ ...prev, backup_checkpoints: checked }))
                    }
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="ui" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Interface Settings</CardTitle>
              <CardDescription>
                Customize the user interface appearance and behavior
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-2 gap-6">
                <div className="space-y-2">
                  <Label htmlFor="theme">Theme</Label>
                  <Select value={settings.theme} onValueChange={(value) => 
                    setSettings(prev => ({ ...prev, theme: value }))
                  }>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="light">Light Theme</SelectItem>
                      <SelectItem value="dark">Dark Theme</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="language">Language</Label>
                  <Select value={settings.language} onValueChange={(value) => 
                    setSettings(prev => ({ ...prev, language: value }))
                  }>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="en">English</SelectItem>
                      <SelectItem value="th">ไทย (Thai)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="auto-refresh">Auto Refresh Interval (ms)</Label>
                  <Input
                    id="auto-refresh"
                    type="number"
                    value={settings.auto_refresh_interval}
                    onChange={(e) => setSettings(prev => ({ 
                      ...prev, 
                      auto_refresh_interval: parseInt(e.target.value) 
                    }))}
                    min="1000"
                    max="60000"
                    step="1000"
                  />
                  <p className="text-xs text-muted-foreground">
                    How often to refresh data automatically
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="thumbnail-size">Thumbnail Size (px)</Label>
                  <Input
                    id="thumbnail-size"
                    type="number"
                    value={settings.thumbnail_size}
                    onChange={(e) => setSettings(prev => ({ 
                      ...prev, 
                      thumbnail_size: parseInt(e.target.value) 
                    }))}
                    min="100"
                    max="500"
                    step="50"
                  />
                  <p className="text-xs text-muted-foreground">
                    Size of image thumbnails in pixels
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="items-per-page">Items Per Page</Label>
                  <Input
                    id="items-per-page"
                    type="number"
                    value={settings.items_per_page}
                    onChange={(e) => setSettings(prev => ({ 
                      ...prev, 
                      items_per_page: parseInt(e.target.value) 
                    }))}
                    min="10"
                    max="200"
                    step="10"
                  />
                  <p className="text-xs text-muted-foreground">
                    Number of items to show per page in lists
                  </p>
                </div>
              </div>

              <Separator />

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Show Confidence Scores</Label>
                  <p className="text-sm text-muted-foreground">
                    Display AI confidence scores in results and reviews
                  </p>
                </div>
                <Switch
                  checked={settings.show_confidence_scores}
                  onCheckedChange={(checked) => 
                    setSettings(prev => ({ ...prev, show_confidence_scores: checked }))
                  }
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="health" className="space-y-6">
          {systemHealth && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  {getHealthStatusIcon(systemHealth.status)}
                  System Health
                  <Badge variant={systemHealth.status === 'healthy' ? 'default' : 
                                 systemHealth.status === 'warning' ? 'secondary' : 'destructive'}>
                    {systemHealth.status.toUpperCase()}
                  </Badge>
                </CardTitle>
                <CardDescription>
                  Current system performance and resource usage
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <Cpu className="h-4 w-4" />
                      <Label>CPU Usage</Label>
                    </div>
                    <Progress value={systemHealth.cpu_usage_percent} className="h-2" />
                    <p className="text-sm text-muted-foreground">
                      {systemHealth.cpu_usage_percent.toFixed(1)}%
                    </p>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <MemoryStick className="h-4 w-4" />
                      <Label>Memory Usage</Label>
                    </div>
                    <Progress value={systemHealth.memory_usage_percent} className="h-2" />
                    <p className="text-sm text-muted-foreground">
                      {systemHealth.memory_usage_percent.toFixed(1)}% 
                      ({systemHealth.memory_available_gb.toFixed(1)} GB available)
                    </p>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <HardDrive className="h-4 w-4" />
                      <Label>Disk Usage</Label>
                    </div>
                    <Progress value={systemHealth.disk_usage_percent} className="h-2" />
                    <p className="text-sm text-muted-foreground">
                      {systemHealth.disk_usage_percent.toFixed(1)}% 
                      ({systemHealth.disk_free_gb.toFixed(1)} GB free)
                    </p>
                  </div>
                </div>

                {systemHealth.gpu_metrics && systemHealth.gpu_metrics.length > 0 && (
                  <div>
                    <h4 className="font-medium mb-4">GPU Metrics</h4>
                    <div className="space-y-4">
                      {systemHealth.gpu_metrics.map((gpu, index) => (
                        <div key={index} className="border rounded-lg p-4">
                          <h5 className="font-medium mb-2">{gpu.name}</h5>
                          <div className="grid grid-cols-3 gap-4">
                            <div className="space-y-1">
                              <Label className="text-xs">Load</Label>
                              <Progress value={gpu.load} className="h-2" />
                              <p className="text-xs text-muted-foreground">{gpu.load.toFixed(1)}%</p>
                            </div>
                            <div className="space-y-1">
                              <Label className="text-xs">Memory</Label>
                              <Progress value={gpu.memory_used_percent} className="h-2" />
                              <p className="text-xs text-muted-foreground">{gpu.memory_used_percent.toFixed(1)}%</p>
                            </div>
                            <div className="space-y-1">
                              <Label className="text-xs">Temperature</Label>
                              <p className="text-sm font-medium">{gpu.temperature}°C</p>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {hardwareInfo && (
            <Card>
              <CardHeader>
                <CardTitle>Hardware Information</CardTitle>
                <CardDescription>
                  Detailed information about your system hardware
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label className="text-sm font-medium">Platform</Label>
                    <p className="text-sm text-muted-foreground">{hardwareInfo.platform}</p>
                  </div>
                  <div>
                    <Label className="text-sm font-medium">Processor</Label>
                    <p className="text-sm text-muted-foreground">{hardwareInfo.processor}</p>
                  </div>
                  <div>
                    <Label className="text-sm font-medium">CPU Cores</Label>
                    <p className="text-sm text-muted-foreground">
                      {hardwareInfo.cpu_count_physical} physical, {hardwareInfo.cpu_count_logical} logical
                    </p>
                  </div>
                  <div>
                    <Label className="text-sm font-medium">Total Memory</Label>
                    <p className="text-sm text-muted-foreground">{hardwareInfo.memory_total_gb.toFixed(1)} GB</p>
                  </div>
                  {hardwareInfo.gpus.length > 0 && (
                    <div className="col-span-2">
                      <Label className="text-sm font-medium">GPUs</Label>
                      <div className="space-y-1">
                        {hardwareInfo.gpus.map((gpu, index) => (
                          <p key={index} className="text-sm text-muted-foreground">
                            {gpu.name} ({gpu.memory_total} MB VRAM)
                          </p>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="advanced" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Configuration Management</CardTitle>
              <CardDescription>
                Import, export, and manage your configuration settings
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-2 gap-4">
                <Button variant="outline" onClick={exportSettings}>
                  <Download className="h-4 w-4 mr-2" />
                  Export Configuration
                </Button>
                <Button variant="outline">
                  <Upload className="h-4 w-4 mr-2" />
                  Import Configuration
                </Button>
                <Button variant="outline" onClick={handleOptimize} disabled={optimizing}>
                  {optimizing ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Zap className="h-4 w-4 mr-2" />
                  )}
                  Hardware Optimization
                </Button>
                <Button variant="destructive" onClick={handleReset} disabled={loading}>
                  <RotateCcw className="h-4 w-4 mr-2" />
                  Reset All Settings
                </Button>
              </div>

              <Separator />

              <Alert>
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  Advanced settings can significantly impact system performance. 
                  Make sure you understand the implications before making changes.
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}