# Task 3: contract-version-diff corpus (bounded licence fallback)

## Delivered scope

**Two real, original, English PDF documents; one proven version pair: EUPL 1.1 → 1.2.** 13 physical pages; 375,106 original bytes. These are public standard licence texts, not signed vendor contracts. No synthetic source documents, fake DOCX conversions, scans or private documents. DOCX preference is not fulfilled.

**40 authored gold cases:** 16 changed-scope-or-wording; 16 unchanged; 8 not-in-corpus. Questions/answers are evaluation annotations, not source documents. No claim of 40 independent provisions or 40 substantive legal amendments: some cases inspect different properties of the same clause, and wording changes may be editorial. Use family/clause-group holdouts rather than random question splits to avoid leakage. All data belongs to one licence family.

### Actual coverage and gaps

- Covered: scope/definition changes; notice wording; copyleft version-only example; compatibility base; additional-agreement wording; appendix-update power; jurisdiction and applicable-law qualifiers; EU-institution role scope; treaty reference; compatible-licence version additions. Includes semantic-qualifier and “do not infer prohibition from deleted wording” traps.
- Covered: matching evidence for retained obligations/rights. Unchanged means the particular proposition is retained, not that the entire clause is byte-identical. Typography, capitalization and conjunctions can differ.
- **Not covered as authentic changes:** commercial amounts, invoice dates, signed execution dates, named customer/vendor replacements, negotiated SLA changes, GST, employment terms. Treaty article numbers are not amounts; publication dates are not execution dates; EU institution categories are not swapped contracting customers. Royalty-free is retained, not a monetary change. Modification-date notices are retained obligations, not changed transaction dates. Those missing categories have scoped abstention traps only.
- No DOCX layout, OCR, scans or cross-family conflict benchmark. Corpus is a bounded, legally authentic fallback, not a representative vendor-contract benchmark. More authentic commercial-version originals are needed before claiming broad Task 3 coverage.

## Layout and schema

- `documents/`: only two unchanged downloaded PDFs.
- `extracted/*.txt`: local Poppler `pdftotext -layout` output, with form-feed physical-page separators retained.
- `extracted/*.pages.json`: extracted text with explicit 1-based physical page numbers; PDF printed page labels are NOT physical page indexes. For 1.2, decision = physical p1; licence annex = physical pp2–6 (printed OJ pp60–64).
- `extracted/gold_clauses.json`: exact normalized quote records tied to file, page, clause and gold case. Not a full independent structural parse.
- `gold.csv`: Task1-compatible first eight columns (`id, question, expected_answer, source_file, page_clause_cell, type, evidence_quote, verification`) plus separate old/new file/page/clause/quote columns, scope and notes. Do not split the combined evidence field when the explicit columns suffice.
- `SOURCES.md`: exact URLs, explicit lineage, rights caveats, source-search exclusions.
- `validation.json`: bytes, SHA-256, type, page count, local open/parse/render checks and warnings per original.
- `audit.json`, `download_log.json`: bounded validation findings and original download provenance.
- `build_dataset.py`: local utility used to author/validate dataset; not application code. Requires Python 3 and Poppler; reads existing originals and extracted text; no models, network or cloud calls. It regenerates gold/validation/page JSON/audit, not this README or auxiliary clause/download records.

## Evidence and abstention rules

For each of 32 answerable cases, quote **both** versions on the stated physical page and clause; 64 whitespace-normalized quote checks pass. Do not equate matching quotation/value strings with validated legal interpretation. Case-specific answers were curator text-reviewed; independent legal review is still needed for consequential use. No QA system accuracy has been measured.

Absent cases intentionally leave quote and precise supporting-page fields empty: there is no honest affirmative supporting quotation for absence. `coverage` and the scope locator instead identify every searched physical page. Each answer uses **“I could not find supporting evidence in the indexed text”** plus complete corpus coverage. Never elevate that to a claim about unseen documents or all possible contracts. Supplemental literal no-match searches are only checks, not semantic proofs of absence.

## Measured local artifact checks

- Both PDFs opened with `pdfinfo`; extracted with `pdftotext -layout`.
- All 13 pages rendered locally using `pdftoppm`; no render failures. Temporary thumbnails discarded, not external uploads.
- All 64 answerable quotation/page checks pass; 8 absent-case supplemental term checks pass.
- EUPL 1.1 emits `Syntax Warning: Unknown Metadata type: '???'`; this is recorded, not concealed. It does not prevent parsing/rendering.
- No models downloaded, no app code modified, no Mac access, no messages sent, no content uploaded to a model/parser API. Only this assigned output directory was written.

Read against the Techkriti shared brief's legal-user version-comparison and exact-evidence requirements. Read Task1 schema for compatibility; Task1 and PLAN were not modified.
