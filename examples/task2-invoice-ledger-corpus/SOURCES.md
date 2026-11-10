# TASK2 invoice-to-ledger: sources

Accessed 9 November 2026 (IST) with plain HTTPS GET from the box. No cloud parsing. No connector uploads. No models.

## Provenance and status
All 15 files come from the public NAAC Self-Study evidence directory of **Swami Keshvanand Institute of Technology, Management & Gramothan (SKIT), Jaipur**, at `https://naac.skit.ac.in/QIF/` (open Apache directory listings; server modification dates in 2021).
- **REAL documents**: organisational procurement records published for accreditation. They are not samples, not generated, and not modified (byte copies; SHA256 values in validation.json).
- **The ledger is a REAL public institutional XLSX.** `4.2.2&4.2.3.xlsx` is the institution's own NAAC data template. It lists library expenditure per year and category (INR lakhs), and its "View Document" cells hyperlink to the invoice/receipt PDFs below. This pairing comes from the publisher, not from us. Caveat: the ledger holds annual category totals, not one row per invoice, so many matches are sums over several documents.
- `1_6.2.3.xlsx` is a real vendor/implementation-year list (no amounts). It is paired with the real `Bills_of_Expenditure.pdf`.
- **No DERIVED or SYNTHETIC ledger was created.** Every mismatch in gold.csv is present in the published documents.

| Local file | Original URL | Kind | Ledger link |
|---|---|---|---|
| skit_library_ledger_2016-21.xlsx | https://naac.skit.ac.in/QIF/4/4.2/4.2.2/4.2.2&4.2.3.xlsx | XLSX ledger (real) | n/a |
| skit_egov_vendor_years.xlsx | https://naac.skit.ac.in/QIF/6/6.2/6.2.3/1_6.2.3.xlsx | XLSX vendor/year list (real) | n/a |
| skit_egov_bills.pdf | https://naac.skit.ac.in/QIF/6/6.2/6.2.3/Bills_of_Expenditure.pdf | 26 pp; mixed text-layer and image pages; contains repeated invoices | 6.2.3 sheet |
| lib_2016-17_ejournal.pdf | …/Subscription_Details/2016-17/e-Journal_Subscription.pdf | scan | F48 |
| lib_2017-18_ejournal.pdf | …/2017-18/e-Journal_Subscription.pdf | scan | F38 |
| lib_2019-20_ejournal.pdf | …/2019-20/e-Journal_Subscription.pdf | scan | F18 |
| lib_2020-21_ejournal.pdf | …/2020-21/e-Journal_Subscription.pdf | scan | F8 |
| lib_2016-17_delnet.pdf | …/2016-17/Database_DELNET.pdf | scan | F52 |
| lib_2017-18_delnet.pdf | …/2017-18/Database_DELNET_merged.pdf | scan | F42 |
| lib_2018-19_delnet.pdf | …/2018-19/Database-DELNET_subscription.pdf | scan | F32 |
| lib_2019-20_delnet.pdf | …/2019-20/Database-DELNET_subscription.pdf | scan | F22 |
| lib_2020-21_delnet.pdf | …/2020-21/Database-DELNET_subscription.pdf | scan | F12 |
| lib_2018-19_ebooks.pdf | …/2018-19/e-Books_Subscription.pdf | scan | F29 |
| lib_2020-21_ebooks.pdf | …/2020-21/e-Books_Subscription.pdf | scan + printed bank advices | F9 |
| lib_2020-21_books.pdf | …/2020-21/Books_Subscription.pdf | scan with handwriting | F6 |

(… = https://naac.skit.ac.in/QIF/4/4.2/4.2.2/Subscription_Details)

## Rights and privacy caveats
- Publicly posted by the institution for NAAC accreditation. No explicit licence was found, so assume all rights reserved. Use for internal evaluation only and do not redistribute.
- The documents contain business bank account numbers, IFSC codes, cheque images, GSTIN/PAN, staff names and signatures, and vendor phone numbers and emails. All are organisational and were published by the institution. No personal consumer data was sought. Do not republish the files or extracted text outside the project.
- GSTINs and PANs are quoted only where needed for vendor-identity gold.

## Reviewed but not included
- Visva-Bharati RTI suo-motu GeM invoice scan (https://www.visvabharati.ac.in/file/Central%20Purchase%20Section.pdf): real, but no paired ledger. Downloaded for triage, then deleted.
- Cambridgeshire County Council FOI invoices paired with its "payments over £500" dataset (WhatDoTheyKnow request_for_invoices_13): a strong real pairing, but the site returned a Cloudflare challenge to curl and timed out for WebFetch. Not acquired, and not Indian GST.
- Other SKIT library PDFs (Books 2016-20, Journals, e-ShodhSindhu, Remote access; 3.5-38 MB each) were not downloaded, to keep scope bounded.
