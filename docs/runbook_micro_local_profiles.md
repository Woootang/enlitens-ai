# Runbook: Micro-Local Profile Operations

This runbook supports the end-to-end lifecycle for micro-local profiles that inform localized messaging, outreach, and research prioritization. Use it whenever you are preparing profiles for deployment, auditing an existing batch, or responding to operational incidents.

---

## 1. Setup Workflow Overview

| Phase | Objective | Primary Owners | Tooling |
| --- | --- | --- | --- |
| Data Preparation | Assemble, validate, and normalize raw locality insights. | Data Ops, Research Leads | `process_multi_agent_corpus.py`, Golden Dataset sheets |
| Environment Configuration | Ensure runtime services reflect the latest profile metadata. | Platform Engineering | `start_processing.sh`, monitoring server, `.env` secrets |
| Monitoring & Validation | Track processing health and content safety checks. | Monitoring Team | `monitor_processing.py`, dashboards in `monitoring_ui/` |
| Review & Release | Validate quality, authorize publication, stage go-live. | Review Council | Docs in `/docs`, change-control templates |

---

## 2. Detailed Setup Steps

### 2.1 Data Preparation
1. **Ingest Source Material**
   - Pull locality notes, survey snippets, agricultural extension reports, and historical outreach transcripts into the staging area (`data/micro_local_raw/`).
   - Confirm each record includes locality identifiers (county, township, postal code) and a timestamp.
2. **Normalize & Tag**
   - Run the corpus processor: `python process_multi_agent_corpus.py --dataset micro_local_raw --out data/micro_local_curated/`.
   - Apply schema mapping to align segments with profile fields (demographics, key concerns, trusted messengers).
   - Tag fictional elements explicitly with `fictional_disclaimer: true`.
3. **Quality Gate**
   - Spot check 10% of entries for missing disclaimers, mismatched localities, and outdated references (>18 months old).
   - Log findings in the "Micro-Local QA" tracker.

### 2.2 Environment Configuration
1. **Secrets & Config Files**
   - Update `.env.micro_local` with access tokens for survey APIs, local news feeds, and monitoring webhooks.
   - Verify secrets rotation dates and document in the change log.
2. **Service Alignment**
   - Restart the processing pipeline: `./start_processing.sh --profile-set micro_local`.
   - Confirm the monitoring server references the `micro_local` namespace in `monitoring_server_enhanced.py`.
3. **Data Availability Checks**
   - Confirm the curated dataset appears in `enlitens_corpus/` with the correct version stamp (e.g., `micro_local_vYYMMDD`).
   - Run smoke tests: `pytest tests/test_dashboard.py::test_micro_local_feed_health` (ensure test exists before running).

### 2.3 Monitoring Checks
1. **Processing Health**
   - Use `python monitor_processing.py --namespace micro_local` to verify recent job statuses.
   - Inspect queue depth and latency metrics in the monitoring UI (`monitoring_ui/` dashboard "Micro-Local Throughput").
2. **Content Safety & Compliance**
   - Ensure disclaimers appear in random profile samples rendered in the UI preview.
   - Review automated classification flags for high-sensitivity locales (tribal lands, small towns <2k population).
3. **Alert Configuration**
   - Confirm alerts route to the #micro-local-ops Slack channel and on-call rotation.
   - Test escalation webhook by triggering a synthetic warning (`python monitoring_server.py --test-alert micro_local`).

---

## 3. Review Protocols

1. **Peer Review**
   - Assign two reviewers: one domain expert (ag policy, rural outreach) and one editorial standards lead.
   - Require reviewers to initial the QA tracker entries for each sampled profile.
2. **Checklist Validation**
   - Use the checklists in Section 5 to confirm readiness prior to sign-off.
   - Document deviations and compensating controls within the release notes.
3. **Release Authorization**
   - Convene a 15-minute sync with Stakeholder Liaison, Compliance, and Monitoring leads.
   - Capture final go/no-go decision in the "Micro-Local Release Register" doc with timestamps.

---

## 4. Escalation Paths

| Severity | Example Triggers | First Responder | Escalation | SLA |
| --- | --- | --- | --- | --- |
| Sev 1 | Missing disclaimers in published profiles, incorrect locality assignments in >10% of sample. | On-call Content Safety | Director of Research & Legal Counsel | Acknowledge 15 min, resolve 2 hrs |
| Sev 2 | Pipeline delay >6 hrs, monitoring outage, incomplete rural representation. | On-call Platform Engineer | Head of Platform, Ops Manager | Acknowledge 30 min, resolve 6 hrs |
| Sev 3 | Minor copy errors, outdated stats <6 months. | Content Editor | Product Owner | Acknowledge 1 hr, resolve next release |
| Sev 4 | Improvement suggestions, wording refinements. | Assigned Reviewer | Backlog Grooming | Review during weekly meeting |

Escalation contact order: On-call (PagerDuty) → Slack #micro-local-ops → Director of Research → Legal Counsel (if disclaimers missing) → Executive Sponsor.

---

## 5. Release Readiness Checklists

### 5.1 Fictional Disclaimer Verification
- [ ] Every profile contains an explicit fictional disclaimer block.
- [ ] Disclaimer language references the relevant locality and clarifies representational intent.
- [ ] Automated scans confirm no placeholder text (e.g., `{{INSERT DISCLAIMER}}`).
- [ ] Legal reviewed any custom disclaimer variants this cycle.

### 5.2 Locality Diversity Coverage
- [ ] Profiles span urban, peri-urban, and rural classifications per regional plan.
- [ ] Each state/territory in scope has ≥2 localities represented.
- [ ] Rural coverage prioritizes counties with historic under-engagement.
- [ ] Key demographic cohorts (age, occupation, cultural communities) have dedicated insights.

### 5.3 Research Coverage Depth
- [ ] Each profile cites ≥3 validated sources (survey, academic, local reporting).
- [ ] Trusted messenger insights validated by at least one qualitative interview.
- [ ] Emerging trend sections updated within the last 90 days.
- [ ] Profile includes mitigation strategies for misinformation themes.

### 5.4 Prediction-Error Usefulness
- [ ] Forecast sections include confidence intervals and rationale.
- [ ] Error analysis references previous campaign outcomes or field tests.
- [ ] Action recommendations articulate fallback options for high-variance predictions.
- [ ] Monitoring plan lists data points to evaluate post-launch accuracy.

---

## 6. Engaging Skeptical and Rural Audiences

1. **Validate Lived Experience**
   - Lead with acknowledgement of local challenges (e.g., crop yield volatility, broadband gaps) before proposing solutions.
   - Reuse quotes from trusted messengers identified in the profile to signal authentic alignment.
2. **Emphasize Practical Outcomes**
   - Highlight track records of programs that delivered measurable benefits in similar communities.
   - Use plain language, avoiding jargon; refer to the "Messaging Plain Talk" section in each profile.
3. **Offer Verification Paths**
   - Provide links or contacts for independent verification (extension offices, cooperative boards).
   - Encourage community feedback loops (e.g., listening sessions) and advertise the schedule.
4. **Respect Autonomy**
   - Frame recommendations as options to evaluate, not mandates, emphasizing local choice and adaptation.
   - Reinforce that fictionalized personas are composites for exploration, not substitutes for community voices.
5. **Consistency in Tone**
   - Adopt steady, respectful validation language such as: “Local growers told us…”, “We verified with county-led data…”, “Neighbors in the co-op emphasized…”.
   - Avoid dismissive phrasing; ensure any counterpoints are backed by data relevant to the locality.

---

## 7. Continuous Improvement

- **Post-Launch Reviews**: Within 30 days of each release, compile outreach performance metrics and compare against prediction expectations.
- **Feedback Intake**: Route community feedback and internal suggestions to the Micro-Local feedback board; categorize by theme and severity.
- **Runbook Updates**: Reassess this document quarterly; include learnings from escalations and reviewer retrospectives.

---

_Last updated: 2025-11-05_
