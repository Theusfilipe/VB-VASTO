"use client";

import { useEffect, useRef, useState } from "react";
import { API_URL, type PolicyCatalog } from "@/lib/types";

async function request<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    cache: "no-store",
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `Backend returned ${response.status}`);
  }
  return response.json() as Promise<T>;
}

function defaultName(policy: string): string {
  return policy.split("/").pop()?.replace(/\.zip$/i, "") ?? policy;
}

export function PolicyManager() {
  const [catalog, setCatalog] = useState<PolicyCatalog | null>(null);
  const [names, setNames] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);
  const [savingPolicy, setSavingPolicy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [policyToDelete, setPolicyToDelete] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const deleteDialog = useRef<HTMLDialogElement>(null);
  const heading = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    let active = true;
    void request<PolicyCatalog>("/simulation/policies").then((next) => {
      if (!active) return;
      setCatalog(next);
      setNames(Object.fromEntries(
        next.policies.map((policy) => [policy, next.names[policy] ?? defaultName(policy)]),
      ));
      setError(null);
    }).catch((reason: unknown) => {
      if (active) setError(reason instanceof Error ? reason.message : "Could not list saved policies.");
    }).finally(() => {
      if (active) setLoading(false);
    });
    return () => { active = false; };
  }, []);

  async function saveName(policy: string) {
    setSavingPolicy(policy);
    setError(null);
    setNotice(null);
    try {
      const next = await request<PolicyCatalog>("/simulation/policies/name", {
        model_name: policy,
        display_name: names[policy]?.trim() ?? "",
      });
      setCatalog(next);
      setNames((current) => ({
        ...current,
        [policy]: next.names[policy] ?? defaultName(policy),
      }));
      setNotice(`Saved name for ${policy}.`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not save the policy name.");
    } finally {
      setSavingPolicy(null);
    }
  }

  function confirmDelete(policy: string) {
    setPolicyToDelete(policy);
    setDeleteError(null);
    deleteDialog.current?.showModal();
  }

  async function deletePolicy() {
    if (!policyToDelete || deleting) return;
    const policy = policyToDelete;
    setDeleting(true);
    setDeleteError(null);
    setNotice(null);
    try {
      await request("/simulation/policies/delete", { model_name: policy });
      setCatalog((current) => current && {
        ...current,
        policies: current.policies.filter((item) => item !== policy),
      });
      setNotice(`Deleted ${policy}.`);
      deleteDialog.current?.close();
      heading.current?.focus();
    } catch (reason) {
      setDeleteError(reason instanceof Error ? reason.message : "Could not delete the policy.");
    } finally {
      setDeleting(false);
    }
  }

  return (
    <main className="dashboard-shell">
      <section className="page-intro">
        <div>
          <p className="eyebrow">POLICY LIBRARY <span>•</span> SAVED CHECKPOINTS</p>
          <h1 ref={heading} tabIndex={-1}>Manage policies</h1>
          <p className="intro-copy">Name, rename, or delete saved policy checkpoints.</p>
        </div>
      </section>

      {error && <div className="error-banner" role="alert"><span>Policy library</span>{error}</div>}
      {notice && <p className="success-message" role="status">{notice}</p>}
      {loading ? <div className="panel policy-empty">Loading saved policies…</div>
        : catalog === null ? null
        : catalog.configured === false ? (
          <div className="panel policy-empty">
            Set <code>ADAPTIVE_AI_MODELS_DIR</code> on the backend to manage saved policies.
          </div>
        ) : catalog?.policies.length ? (
          <section className="policy-list" aria-label="Saved policies">
            {catalog.policies.map((policy) => (
              <form className="panel policy-card" key={policy} onSubmit={(event) => {
                event.preventDefault();
                void saveName(policy);
              }}>
                <div className="policy-card-heading">
                  <div>
                    <p className="section-label">POLICY NAME</p>
                    <h2>{names[policy] || defaultName(policy)}</h2>
                  </div>
                  <code title={policy}>{policy}</code>
                </div>
                <div className="control-row">
                  <label className="policy-name-field">
                    Name
                    <input
                      aria-label={`Name for ${policy}`}
                      maxLength={80}
                      required
                      value={names[policy] ?? ""}
                      onChange={(event) => setNames((current) => ({ ...current, [policy]: event.target.value }))}
                    />
                  </label>
                  <button className="button button-primary" type="submit" disabled={savingPolicy !== null}>
                    {savingPolicy === policy ? "Saving…" : "Save name"}
                  </button>
                  <button className="button button-delete" type="button" disabled={savingPolicy !== null} onClick={() => confirmDelete(policy)}>
                    Delete
                  </button>
                </div>
              </form>
            ))}
          </section>
        ) : (
          <div className="panel policy-empty">
            No saved policies found. Train a policy or place a checkpoint and its <code>config.json</code> in the configured models directory.
          </div>
        )}
      <dialog
        ref={deleteDialog}
        className="panel policy-dialog"
        aria-labelledby="delete-policy-heading"
        aria-describedby="delete-policy-description"
        onCancel={(event) => { if (deleting) event.preventDefault(); }}
        onClose={() => setPolicyToDelete(null)}
      >
        <h2 id="delete-policy-heading">Delete policy?</h2>
        <p id="delete-policy-description">
          This permanently deletes the checkpoint for <strong>{policyToDelete && (catalog?.names[policyToDelete] ?? defaultName(policyToDelete))}</strong>.
          Other checkpoints and their shared configuration will not be deleted.
        </p>
        <code>{policyToDelete}</code>
        {deleteError && <p className="policy-delete-error" role="alert">{deleteError}</p>}
        <div className="control-row">
          <button autoFocus className="button button-outline" type="button" disabled={deleting} onClick={() => deleteDialog.current?.close()}>Cancel</button>
          <button className="button button-danger" type="button" disabled={deleting} onClick={() => void deletePolicy()}>
            {deleting ? "Deleting…" : "Delete policy"}
          </button>
        </div>
      </dialog>
    </main>
  );
}
