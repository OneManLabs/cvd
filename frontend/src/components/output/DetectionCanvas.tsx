import { useRef, useEffect } from 'react';
import type { InferenceResult } from '../../lib/api';
import { getClassColor } from '../../lib/colors';

interface DetectionCanvasProps {
  results: InferenceResult[];
  width: number;  // source video/image width
  height: number; // source video/image height
}

export function DetectionCanvas({ results, width, height }: DetectionCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const containerW = canvas.clientWidth;
    const containerH = canvas.clientHeight;

    canvas.width = containerW;
    canvas.height = containerH;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, containerW, containerH);

    // Calculate the actual video display area (object-contain letterboxing)
    const sourceAspect = width / height;
    const containerAspect = containerW / containerH;

    let drawW: number, drawH: number, offsetX: number, offsetY: number;

    if (sourceAspect > containerAspect) {
      // Video is wider — black bars top/bottom
      drawW = containerW;
      drawH = containerW / sourceAspect;
      offsetX = 0;
      offsetY = (containerH - drawH) / 2;
    } else {
      // Video is taller — black bars left/right
      drawH = containerH;
      drawW = containerH * sourceAspect;
      offsetX = (containerW - drawW) / 2;
      offsetY = 0;
    }

    for (const result of results) {
      for (const det of result.detections) {
        const color = getClassColor(det.class_id);
        const [x1, y1, x2, y2] = det.bbox;
        const rx = offsetX + x1 * drawW;
        const ry = offsetY + y1 * drawH;
        const rw = (x2 - x1) * drawW;
        const rh = (y2 - y1) * drawH;

        // Draw segmentation mask
        if (det.mask && det.mask.length > 0) {
          ctx.beginPath();
          ctx.moveTo(offsetX + det.mask[0][0] * drawW, offsetY + det.mask[0][1] * drawH);
          for (let i = 1; i < det.mask.length; i++) {
            ctx.lineTo(offsetX + det.mask[i][0] * drawW, offsetY + det.mask[i][1] * drawH);
          }
          ctx.closePath();
          ctx.fillStyle = color + '25';
          ctx.fill();
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.stroke();
        }

        // Draw bounding box (only if no segmentation mask)
        if (!det.mask || det.mask.length === 0) {
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.strokeRect(rx, ry, rw, rh);
        }

        // Draw label background
        const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = '12px Helvetica, Arial, sans-serif';
        const textWidth = ctx.measureText(label).width;
        const labelHeight = 18;
        const labelY = ry > labelHeight ? ry - labelHeight : ry;
        ctx.fillStyle = color;
        ctx.fillRect(rx, labelY, textWidth + 8, labelHeight);
        ctx.fillStyle = '#fff';
        ctx.fillText(label, rx + 4, labelY + 13);

        // Draw pose keypoints
        if (det.keypoints && det.keypoints.length > 0) {
          for (const kpt of det.keypoints) {
            if (kpt.length >= 2 && (kpt[0] > 0 || kpt[1] > 0)) {
              ctx.beginPath();
              ctx.arc(offsetX + kpt[0] * drawW, offsetY + kpt[1] * drawH, 3, 0, Math.PI * 2);
              ctx.fillStyle = color;
              ctx.fill();
            }
          }

          // Draw skeleton connections (COCO format)
          const skeleton = [
            [0, 1], [0, 2], [1, 3], [2, 4],
            [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
            [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
          ];
          ctx.strokeStyle = color + '80';
          ctx.lineWidth = 1.5;
          for (const [a, b] of skeleton) {
            if (a < det.keypoints.length && b < det.keypoints.length) {
              const ka = det.keypoints[a];
              const kb = det.keypoints[b];
              if (ka[0] > 0 && ka[1] > 0 && kb[0] > 0 && kb[1] > 0) {
                ctx.beginPath();
                ctx.moveTo(offsetX + ka[0] * drawW, offsetY + ka[1] * drawH);
                ctx.lineTo(offsetX + kb[0] * drawW, offsetY + kb[1] * drawH);
                ctx.stroke();
              }
            }
          }
        }
      }
    }
  }, [results, width, height]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none"
    />
  );
}
