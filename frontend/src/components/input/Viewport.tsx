import { useRef, useEffect, useCallback, useState } from "react";
import { Camera, CameraOff, ImageIcon, Video, Upload, Play, Square, X } from "lucide-react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "../ui/Tabs";
import { Button } from "../ui/Button";
import { cn } from "../../lib/utils";
import type { SourceType } from "../../lib/constants";
import type { InferenceResult } from "../../lib/api";
import { DetectionCanvas } from "../output/DetectionCanvas";

interface ViewportProps {
  source: SourceType;
  onSourceChange: (source: SourceType) => void;
  isRunning: boolean;
  onToggleRun: () => void;
  onFrame: (canvas: HTMLCanvasElement) => void;
  onImageSelected: (file: File, preview: string) => void;
  imagePreview: string | null;
  onClearImage: () => void;
  results: InferenceResult[];
  canvasSize: { width: number; height: number };
  activeModelCount: number;
}

export function Viewport({
  source,
  onSourceChange,
  isRunning,
  onToggleRun,
  onFrame,
  onImageSelected,
  imagePreview,
  onClearImage,
  results,
  canvasSize,
  activeModelCount,
}: ViewportProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const hiddenCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number>(0);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  // Video upload state
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const uploadVideoRef = useRef<HTMLVideoElement>(null);
  const uploadCanvasRef = useRef<HTMLCanvasElement>(null);
  const videoRafRef = useRef<number>(0);

  // Start webcam
  useEffect(() => {
    if (source !== "webcam") return;
    let cancelled = false;
    const videoElement = videoRef.current;

    const init = async () => {
      try {
        setCameraError(null);
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 } },
        });
        if (cancelled) { stream.getTracks().forEach((t) => t.stop()); return; }
        streamRef.current = stream;
        if (videoElement) {
          videoElement.srcObject = stream;
          await videoElement.play();
        }
      } catch {
        if (!cancelled) setCameraError("Could not access camera.");
      }
    };
    init();

    return () => {
      cancelled = true;
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
      if (videoElement) videoElement.srcObject = null;
      cancelAnimationFrame(rafRef.current);
    };
  }, [source]);

  // Webcam frame capture
  useEffect(() => {
    if (source !== "webcam" || !isRunning) return;
    let lastTime = 0;
    const loop = (time: number) => {
      if (time - lastTime > 66) {
        lastTime = time;
        const video = videoRef.current;
        const canvas = hiddenCanvasRef.current;
        if (video && canvas && video.readyState >= 2) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          const ctx = canvas.getContext("2d");
          if (ctx) { ctx.drawImage(video, 0, 0); onFrame(canvas); }
        }
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [source, isRunning, onFrame]);

  // Video upload frame capture
  useEffect(() => {
    if (source !== "video" || !isRunning || !videoUrl) return;
    const video = uploadVideoRef.current;
    if (video?.paused) video.play();

    let lastTime = 0;
    const loop = (time: number) => {
      if (time - lastTime > 66) {
        lastTime = time;
        const canvas = uploadCanvasRef.current;
        if (video && canvas && video.readyState >= 2) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          const ctx = canvas.getContext("2d");
          if (ctx) { ctx.drawImage(video, 0, 0); onFrame(canvas); }
        }
      }
      videoRafRef.current = requestAnimationFrame(loop);
    };
    videoRafRef.current = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(videoRafRef.current);
      if (video && !video.paused) video.pause();
    };
  }, [source, isRunning, videoUrl, onFrame]);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (!file) return;
      if (source === "image" && file.type.startsWith("image/")) {
        onImageSelected(file, URL.createObjectURL(file));
      } else if (source === "video" && file.type.startsWith("video/")) {
        setVideoUrl(URL.createObjectURL(file));
      }
    },
    [source, onImageSelected]
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      if (source === "image") {
        onImageSelected(file, URL.createObjectURL(file));
      } else if (source === "video") {
        setVideoUrl(URL.createObjectURL(file));
      }
      e.target.value = "";
    },
    [source, onImageSelected]
  );

  return (
    <div className="cvd-viewport flex flex-1 flex-col">
      <Tabs value={source} onValueChange={(v) => onSourceChange(v as SourceType)} className="flex flex-1 flex-col">
        <div className="viewport-toolbar flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="cvd-eyebrow hidden xl:inline">Input source</span>
            <TabsList className="source-tabs">
            <TabsTrigger value="webcam" className="gap-1.5 text-xs">
              <Camera className="h-3.5 w-3.5" />
              Webcam
            </TabsTrigger>
            <TabsTrigger value="image" className="gap-1.5 text-xs">
              <ImageIcon className="h-3.5 w-3.5" />
              Image
            </TabsTrigger>
            <TabsTrigger value="video" className="gap-1.5 text-xs">
              <Video className="h-3.5 w-3.5" />
              Video
            </TabsTrigger>
            </TabsList>
          </div>

          <div className="flex items-center gap-3">
            {activeModelCount === 0 && (
              <span className="text-xs text-warning">Select a model</span>
            )}
            <Button
              variant={isRunning ? "destructive" : "default"}
              size="sm"
              onClick={onToggleRun}
              className="run-button gap-1.5"
              disabled={activeModelCount === 0}
            >
              {isRunning ? (
                <><Square className="h-3.5 w-3.5" /> Stop</>
              ) : (
                <><Play className="h-3.5 w-3.5" /> Start</>
              )}
            </Button>
          </div>
        </div>

        {/* Webcam */}
        <TabsContent value="webcam" className="mt-0 flex-1">
          <div className="viewport-stage relative flex h-full items-center justify-center">
            {cameraError ? (
              <div className="text-center">
                <CameraOff className="mx-auto h-12 w-12 text-muted-foreground/50" />
                <p className="mt-2 text-sm text-muted-foreground">{cameraError}</p>
              </div>
            ) : (
              <div className="relative w-full h-full">
                <video ref={videoRef} className="w-full h-full object-contain" playsInline muted />
                <canvas ref={hiddenCanvasRef} className="hidden" />
                {results.length > 0 && (
                  <DetectionCanvas results={results} width={canvasSize.width} height={canvasSize.height} />
                )}
              </div>
            )}
          </div>
        </TabsContent>

        {/* Image */}
        <TabsContent value="image" className="mt-0 flex-1">
          <div
            className={cn(
              "viewport-stage relative flex h-full items-center justify-center transition-colors",
              isDragging ? "is-dragging" : ""
            )}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
          >
            {imagePreview ? (
              <div className="relative w-full h-full">
                <img src={imagePreview} alt="Upload" className="w-full h-full object-contain" />
                <button
                  onClick={onClearImage}
                  className="absolute top-3 right-3 flex h-7 w-7 items-center justify-center rounded-lg bg-card/80 border border-border text-muted-foreground hover:text-foreground transition-colors"
                >
                  <X className="h-4 w-4" />
                </button>
                {results.length > 0 && (
                  <DetectionCanvas results={results} width={canvasSize.width} height={canvasSize.height} />
                )}
              </div>
            ) : (
              <label className="upload-zone flex cursor-pointer flex-col items-center gap-4 p-12">
                <div className="upload-zone-icon rounded-full p-4">
                  <Upload className="h-8 w-8 text-muted-foreground" />
                </div>
                <div className="text-center">
                  <p className="font-medium">Drop an image here</p>
                  <p className="text-sm text-muted-foreground">or click to browse</p>
                </div>
                <input type="file" accept="image/*" className="hidden" onChange={handleFileSelect} />
              </label>
            )}
          </div>
        </TabsContent>

        {/* Video */}
        <TabsContent value="video" className="mt-0 flex-1">
          <div
            className={cn(
              "viewport-stage relative flex h-full items-center justify-center transition-colors",
              isDragging ? "is-dragging" : ""
            )}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
          >
            {videoUrl ? (
              <div className="relative w-full h-full">
                <video ref={uploadVideoRef} src={videoUrl} className="w-full h-full object-contain" playsInline loop muted />
                <canvas ref={uploadCanvasRef} className="hidden" />
                <button
                  onClick={() => setVideoUrl(null)}
                  className="absolute top-3 right-3 flex h-7 w-7 items-center justify-center rounded-lg bg-card/80 border border-border text-muted-foreground hover:text-foreground transition-colors"
                >
                  <X className="h-4 w-4" />
                </button>
                {results.length > 0 && (
                  <DetectionCanvas results={results} width={canvasSize.width} height={canvasSize.height} />
                )}
              </div>
            ) : (
              <label className="upload-zone flex cursor-pointer flex-col items-center gap-4 p-12">
                <div className="upload-zone-icon rounded-full p-4">
                  <Video className="h-8 w-8 text-muted-foreground" />
                </div>
                <div className="text-center">
                  <p className="font-medium">Drop a video here</p>
                  <p className="text-sm text-muted-foreground">or click to browse</p>
                </div>
                <input type="file" accept="video/*" className="hidden" onChange={handleFileSelect} />
              </label>
            )}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
