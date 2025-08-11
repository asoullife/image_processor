import { useRouter } from "next/router";

export default function MonitoringPage() {
  const { sessionId } = useRouter().query;
  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold">Monitoring {sessionId}</h1>
      <p className="text-muted-foreground">Real-time monitoring coming soon.</p>
    </div>
  );
}
