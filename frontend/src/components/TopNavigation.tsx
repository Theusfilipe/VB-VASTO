import Link from "next/link";
import type { ReactNode } from "react";

export function TopNavigation({
  current,
  meta,
}: {
  current: "simulation" | "policies";
  meta: ReactNode;
}) {
  return (
    <header className="topbar">
      <Link className="brand" href="/" aria-label="Adaptive Field home">
        <span className="brand-mark">a.</span>
        <span>adaptive<span className="brand-light">field</span></span>
      </Link>
      <nav className="topbar-tabs" aria-label="Main navigation">
        <Link className={current === "simulation" ? "topbar-tab active" : "topbar-tab"} href="/" aria-current={current === "simulation" ? "page" : undefined}>Simulation</Link>
        <Link className={current === "policies" ? "topbar-tab active" : "topbar-tab"} href="/policies" aria-current={current === "policies" ? "page" : undefined}>Policies</Link>
      </nav>
      <div className="topbar-meta">{meta}</div>
    </header>
  );
}
