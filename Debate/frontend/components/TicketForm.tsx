"use client";

import { useMemo, useState } from "react";

import { SectionCard } from "@/components/SectionCard";
import type { IngestFailureTicketRequest, SourceType } from "@/types/api";

type TicketFormProps = {
  on_submit: (payload: IngestFailureTicketRequest) => Promise<void>;
  loading: boolean;
};

type TicketFormState = {
  sourceType: SourceType;
  sourceRef: string;
  title: string;
  rawText: string;
  eventTimestamp: string;
  agentId: string;
  scenarioId: string;
  artifactsRef: string;
};

type FieldName = keyof TicketFormState;
type TicketFormErrors = Partial<Record<FieldName, string>>;

const DEFAULT_STATE: TicketFormState = {
  sourceType: "jira",
  sourceRef: "",
  title: "",
  rawText: "",
  eventTimestamp: new Date().toISOString().slice(0, 16),
  agentId: "",
  scenarioId: "",
  artifactsRef: ""
};

const EXAMPLE_STATE: TicketFormState = {
  sourceType: "jira",
  sourceRef: "JIRA-8421",
  title: "Vehicle drift during urban left turn in glare",
  rawText:
    "At 14:32 UTC, the AV drifted toward the bike lane during an urban left turn after lane markings became low-contrast due to sun glare. Driver intervened after lateral offset increased. Replay and log links attached.",
  eventTimestamp: "2026-03-03T14:32",
  agentId: "agent-01",
  scenarioId: "urban-left-turn-22",
  artifactsRef: "replay://leftturn-8421, log://session-8421"
};

function to_request_payload(form_state: TicketFormState): IngestFailureTicketRequest {
  const artifacts = form_state.artifactsRef
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);

  return {
    sourceType: form_state.sourceType,
    sourceRef: form_state.sourceRef.trim(),
    title: form_state.title.trim(),
    rawText: form_state.rawText.trim(),
    eventTimestamp: new Date(form_state.eventTimestamp).toISOString(),
    agentId: form_state.agentId.trim() ? form_state.agentId.trim() : null,
    scenarioId: form_state.scenarioId.trim() ? form_state.scenarioId.trim() : null,
    artifactsRef: artifacts.length > 0 ? artifacts : null
  };
}

function validate_form_state(form_state: TicketFormState): TicketFormErrors {
  const errors: TicketFormErrors = {};

  if (!form_state.sourceRef.trim()) {
    errors.sourceRef = "Source reference is required.";
  }
  if (!form_state.title.trim()) {
    errors.title = "Title is required.";
  }
  if (!form_state.rawText.trim()) {
    errors.rawText = "Raw ticket text is required.";
  } else if (form_state.rawText.trim().length > 100000) {
    errors.rawText = "Raw ticket text must be under 100,000 characters.";
  }
  if (!form_state.eventTimestamp.trim()) {
    errors.eventTimestamp = "Event timestamp is required.";
  } else if (Number.isNaN(new Date(form_state.eventTimestamp).getTime())) {
    errors.eventTimestamp = "Event timestamp must be a valid date/time.";
  }

  return errors;
}

export function TicketForm({ on_submit, loading }: TicketFormProps): JSX.Element {
  const [form_state, set_form_state] = useState<TicketFormState>(DEFAULT_STATE);
  const [touched_fields, set_touched_fields] = useState<Partial<Record<FieldName, boolean>>>({});
  const [example_loaded, set_example_loaded] = useState<boolean>(false);

  const form_errors = useMemo(() => validate_form_state(form_state), [form_state]);
  const form_valid = useMemo(() => Object.keys(form_errors).length === 0, [form_errors]);

  function update_field<K extends keyof TicketFormState>(key: K, value: TicketFormState[K]): void {
    set_form_state((current) => ({ ...current, [key]: value }));
  }

  function mark_touched(field_name: FieldName): void {
    set_touched_fields((current) => ({ ...current, [field_name]: true }));
  }

  function field_error(field_name: FieldName): string | null {
    if (!touched_fields[field_name]) {
      return null;
    }
    return form_errors[field_name] ?? null;
  }

  function load_example_ticket(): void {
    set_form_state(EXAMPLE_STATE);
    set_touched_fields({});
    set_example_loaded(true);
    setTimeout(() => set_example_loaded(false), 1800);
  }

  async function on_submit_form(event: React.FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    set_touched_fields({
      sourceRef: true,
      title: true,
      rawText: true,
      eventTimestamp: true
    });
    if (!form_valid || loading) {
      return;
    }
    await on_submit(to_request_payload(form_state));
  }

  return (
    <SectionCard
      title="Ticket Context"
      subtitle="Ingest evidence and generate a normalized failure capsule."
    >
      <form className="ticket-form" onSubmit={on_submit_form}>
        <div className="ticket-form-toolbar">
          <button type="button" className="secondary-action" onClick={load_example_ticket}>
            Use Example Ticket
          </button>
          {example_loaded ? <span className="hint-inline">Example ticket loaded</span> : null}
        </div>

        <div className="form-grid">
          <label>
            Source Type
            <select
              value={form_state.sourceType}
              onChange={(event) => update_field("sourceType", event.target.value as SourceType)}
            >
              <option value="jira">jira</option>
              <option value="slack">slack</option>
              <option value="manual">manual</option>
            </select>
          </label>

          <label className={field_error("sourceRef") ? "field-invalid" : ""}>
            Source Ref <span className="required-star">*</span>
            <input
              value={form_state.sourceRef}
              onChange={(event) => update_field("sourceRef", event.target.value)}
              onBlur={() => mark_touched("sourceRef")}
              placeholder="JIRA-123 or thread URL"
              required
            />
            {field_error("sourceRef") ? (
              <span className="field-error">{field_error("sourceRef")}</span>
            ) : null}
          </label>

          <label className={field_error("eventTimestamp") ? "field-invalid" : ""}>
            Event Timestamp <span className="required-star">*</span>
            <input
              type="datetime-local"
              value={form_state.eventTimestamp}
              onChange={(event) => update_field("eventTimestamp", event.target.value)}
              onBlur={() => mark_touched("eventTimestamp")}
              required
            />
            {field_error("eventTimestamp") ? (
              <span className="field-error">{field_error("eventTimestamp")}</span>
            ) : null}
          </label>

          <label>
            Agent ID (optional)
            <input
              value={form_state.agentId}
              onChange={(event) => update_field("agentId", event.target.value)}
              placeholder="agent-01"
            />
          </label>

          <label>
            Scenario ID (optional)
            <input
              value={form_state.scenarioId}
              onChange={(event) => update_field("scenarioId", event.target.value)}
              placeholder="scenario-001"
            />
          </label>

          <label className="full-width">
            Artifacts (optional, comma separated)
            <input
              value={form_state.artifactsRef}
              onChange={(event) => update_field("artifactsRef", event.target.value)}
              placeholder="replay://abc, log://xyz"
            />
          </label>
        </div>

        <label className={field_error("title") ? "field-invalid" : ""}>
          Title <span className="required-star">*</span>
          <input
            value={form_state.title}
            onChange={(event) => update_field("title", event.target.value)}
            onBlur={() => mark_touched("title")}
            placeholder="Vehicle drifted during lane transition"
            required
          />
          {field_error("title") ? <span className="field-error">{field_error("title")}</span> : null}
        </label>

        <label className={field_error("rawText") ? "field-invalid" : ""}>
          Raw Ticket Text <span className="required-star">*</span>
          <textarea
            value={form_state.rawText}
            onChange={(event) => update_field("rawText", event.target.value)}
            onBlur={() => mark_touched("rawText")}
            placeholder="Paste full failure narrative, logs, and context."
            rows={7}
            required
          />
          <div className="field-hint-row">
            <span>{form_state.rawText.trim().length.toLocaleString()} / 100,000 chars</span>
          </div>
          {field_error("rawText") ? <span className="field-error">{field_error("rawText")}</span> : null}
        </label>

        <button type="submit" disabled={!form_valid || loading}>
          {loading ? "Generating Capsule..." : "Ingest and Generate Capsule"}
        </button>
      </form>
    </SectionCard>
  );
}
