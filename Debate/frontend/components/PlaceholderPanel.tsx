import { SectionCard } from "@/components/SectionCard";

type PlaceholderPanelProps = {
  title: string;
  description: string;
};

export function PlaceholderPanel({ title, description }: PlaceholderPanelProps): JSX.Element {
  return (
    <SectionCard title={title}>
      <div className="placeholder-panel">
        <div className="placeholder-badge">Coming Soon</div>
        <p>{description}</p>
        <span>Waiting on backend endpoint implementation.</span>
      </div>
    </SectionCard>
  );
}
