// Distinct colors for detection classes
const CLASS_COLORS = [
  '#6366f1', // indigo
  '#22c55e', // green
  '#f59e0b', // amber
  '#ef4444', // red
  '#06b6d4', // cyan
  '#ec4899', // pink
  '#8b5cf6', // violet
  '#f97316', // orange
  '#14b8a6', // teal
  '#84cc16', // lime
  '#e879f9', // fuchsia
  '#38bdf8', // sky
  '#a3e635', // yellow-green
  '#fb7185', // rose
  '#2dd4bf', // emerald
  '#fbbf24', // yellow
];

export function getClassColor(classId: number): string {
  return CLASS_COLORS[classId % CLASS_COLORS.length];
}
