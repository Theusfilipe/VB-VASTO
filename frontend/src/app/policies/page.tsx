import { PolicyManager } from "@/components/PolicyManager";
import { TopNavigation } from "@/components/TopNavigation";

export default function PoliciesPage() {
  return (
    <>
      <div className="dashboard-shell">
        <TopNavigation current="policies" meta={<>Saved policy library</>} />
      </div>
      <PolicyManager />
    </>
  );
}
