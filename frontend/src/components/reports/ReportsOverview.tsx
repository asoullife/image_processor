export interface ReportsOverviewProps {
  sessionId: string;
  projectName: string;
  onImageSelect?: (id: string) => void;
}

export function ReportsOverview({ sessionId, projectName, onImageSelect }: ReportsOverviewProps) {
  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Reports for {projectName}</h2>
      <p className="text-sm text-muted-foreground">Session: {sessionId}</p>
      <button
        className="mt-4 px-4 py-2 bg-blue-600 text-white rounded"
        onClick={() => onImageSelect?.("example-image")}
      >
        Select Example Image
      </button>
    </div>
  );
}
