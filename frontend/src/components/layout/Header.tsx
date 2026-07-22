import { Cpu, MonitorPlay, GraduationCap, Activity } from "lucide-react";
import { Tabs, TabsList, TabsTrigger } from "../ui/Tabs";
import { cn } from "../../lib/utils";

export type AppMode = "inference" | "training";

interface HeaderProps {
  mode: AppMode;
  onModeChange: (mode: AppMode) => void;
  device: string;
  hardware: "cuda" | "metal" | "cpu";
}

export function Header({ mode, onModeChange, device, hardware }: HeaderProps) {
  return (
    <header className="cvd-header flex items-center justify-between">
      <div className="flex items-center gap-5">
        <div className="cvd-brand flex items-center gap-3">
          <div className="cvd-mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <div>
            <span className="cvd-brand-name">CVD</span>
            <span className="cvd-brand-subtitle">Vision operations</span>
          </div>
        </div>

        <div className="cvd-device hidden sm:flex items-center gap-2.5">
          <Cpu className="h-3.5 w-3.5" />
          <span className="text-xs font-medium">{device}</span>
          <span
            className={cn(
              "cvd-device-dot h-1.5 w-1.5 rounded-full",
              hardware !== "cpu" ? "bg-success" : "bg-warning"
            )}
          />
        </div>
      </div>

      <div className="flex items-center gap-3">
        <div className="cvd-live hidden lg:flex items-center gap-2">
          <Activity className="h-3.5 w-3.5" />
          <span>Workspace ready</span>
        </div>
        <Tabs value={mode} onValueChange={(v) => onModeChange(v as AppMode)}>
          <TabsList className="cvd-mode-tabs">
          <TabsTrigger value="inference" className="gap-1.5 px-4">
            <MonitorPlay className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Inference</span>
          </TabsTrigger>
          <TabsTrigger value="training" className="gap-1.5 px-4">
            <GraduationCap className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Training</span>
          </TabsTrigger>
          </TabsList>
        </Tabs>
      </div>
    </header>
  );
}
