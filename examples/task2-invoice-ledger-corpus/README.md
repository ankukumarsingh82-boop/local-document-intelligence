# TASK2 invoice-to-ledger: real-document corpus and gold (v0.1)

**Status: corpus assembly only. No model, retrieval, or app was run. No model results exist.** The scripts here (`build_dataset.py` for validation/extraction, `make_gold.py` for writing gold.csv) are dataset utilities, not product code.

## Contents
- `files/`: 13 PDFs + 2 XLSX, unmodified byte copies of real public SKIT Jaipur NAAC evidence (see SOURCES.md).
- `validation.json`: SHA256, bytes, MIME type, page counts from pdfinfo and pypdf, per-page text chars, image count, and page class (text_layer / image_only_scan / blank_or_minimal); for XLSX, openpyxl load result, sheets, dimensions, formula count (0).
- `extracted/*.pdf.txt`: pdftotext -layout output per page with `=== PAGE n ===` markers. Most library pages are scans, so this text is near-empty; that is the extraction-gap condition.
- `extracted/*.xlsx.cells.txt`: every non-empty cell as `Sheet!A1: value` plus hyperlink targets.
- `extracted/VISUAL_TRANSCRIPTIONS.md`: values read visually from the scans for gold (not OCR). Human re-check required.
- `gold.csv`: 51 rows.

## gold.csv schema
Starts from the Task1 columns (id, question, expected_answer, source_file, page_clause_cell, evidence_quote). Task1's `type` and `verification` are replaced by `category` and `evidence_basis`. Added: expected_state, ledger_file, ledger_cell (`'sheet'!cell`), ledger_value (raw cell value; ledger unit is **INR lakhs**), derivation (unit conversion / sum / recompute), notes. Pages are 1-based physical PDF pages. Contradictions cite both sides.

## Case counts (51)
| Group | n | Categories |
|---|---|---|
| Match | 15 | match_amount 8, multi-document sum 4, vendor-year 2, arithmetic 1 |
| Amount mismatch / tolerance | 6 | amount_mismatch 4 (3 printed bill vs recorded net; 1 quotation vs invoice), amount_tolerance 2 (rounding within Re 1) |
| GST / tax | 11 | gst_check 6, gst_rounding 1, gst_exempt 1, gst_unresolved 1, tax_regime (pre-GST service tax) 2 |
| Vendor | 5 | GSTIN change under same PAN, name spelling variants, supplier-code collision, vendor-of-record ≠ publisher in ledger ×2 |
| Date / period | 7 | FY mismatch 3, label vs period 1, ambiguous DD/MM vs MM/DD 1, order-ref-after-invoice 1, recurring invoice vs implementation year (not comparable) 1 |
| Duplicate | 2 | the same TCS and infoTvariT invoices each repeated 4× in one PDF (render hash identical) |
| Not in corpus | 5 | referenced-but-missing final invoice, blank ledger cells ×2, missing page 1 of 2, receipt without invoice |

The figure for genuine scans is in validation.json: 79 of 99 PDF pages are image-only. All library PDFs are photocopier scans, with stamps, handwriting, punch holes, and some pages rotated 90°. Some image pages in skit_egov_bills.pdf are rasterised digital invoices, not paper scans.

## Gaps and limits (honest)
1. **Granularity:** the real ledger has annual category totals, not per-invoice rows. 4 of the 15 matches are multi-document sums, and per-invoice matching is not directly testable against a ledger row.
2. **Visual gold:** 47 of 51 rows rely on visual reading of scans (no OCR was available, and none was installed, to avoid model data). Handwriting cases (books 2020-21) carry extra risk. All need a second human read.
3. **Single institution and publisher:** one college with recurring vendors. This does not show that results generalise.
4. **No synthetic ledger was made.** Duplicate-payment and vendor-mismatch cases come only from what the real documents contain. There is no case of a real double payment; the duplicates are repeated invoice pages within one document.
5. **Only 2 GST-era tax invoices have text layers** (Syndrome p17-19; partly TCS). Other GST values come from scans.
6. Ledger rows have hyperlinks to documents we did not download (Books 2016-20, Journals, e-ShodhSindhu, Remote access). Those rows have no gold.
7. Several GST treatments (zero-GST OIDAR-style invoices, blank IGST rate) are flagged for review, not judged.
8. Rights: public but unlicensed. Internal evaluation only.
