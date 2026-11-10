# Table rows as evidence units (design note)

Status: proposal only. No code has changed for this. Nothing below loads a model.

## Why

A table value means little without its row-mates: the revised rate beside the
existing rate, or the label beside the amount. Today evidence quotes a table
one cell per item, so those row-mates can be split up or dropped.

The scorer's `anchors_covered` in `eval/metrics.py` needs each gold anchor
inside a single cited quote. If the anchor spans two cells, the release can
only score when one item holds the whole row.

## What the index holds today

These numbers come from the replay copies in `eval/results/pipeline-replay`,
read with read-only SQLite.

- **PDF cells.** `ldi/pdftable.py` already finds cells from pdfium geometry.
  Each `DetectedCell` has a row, a column, and a box. `drafts_from_tables`
  turns them into spans through `_table_drafts` in `ldi/spans.py`. The box is
  dropped: every table span has `bbox = ''`.
- **Spaces and small glyphs are lost in PDF cells.** `_characters` skips
  whitespace glyphs and glyphs under 0.15 pt. `_tokens` and `_ruled_cells` then
  join glyphs with `""`. So the cells read `Rs720/-`, `DAEOMNo.` and
  `05032021`, where the page prints `Rs 720/-`, `DAE OM No.` and `05.03.2021`.
  A verbatim anchor cannot match those cells.
- **A detected row is one printed line, not one table row.** A table row whose
  label wraps over four lines becomes four rows: `ResearchPlant`, `Support`,
  `Allowance` and `(NRPSA)` sit on rows 2 to 5, and the rates sit on rows 4
  and 5.
- **XLSX cells.** `_sheet_drafts` uses the sheet row index, so a row is already
  exact. `_expand_merges` copies a merged value onto the cells it covers.
- **The PDF text layer.** The prose spans are pypdf's lines. Some of them
  already hold a whole printed row verbatim, such as
  `1-2 Rs 720/- 1-2 Rs 900/-`. Others store a table column by column, or wrap
  the last words onto the next line.
- **Evidence.** `_table_unit` in `ldi/evidence.py` already groups cells by
  `line` and sorts them by `col`. Then `_span_lines` emits each cell as its own
  source line, and `_evidence_from_source` makes each one an item.

Measured on the 72 dev and held-out replay cases, after Prompts 1 to 4:

| | |
|---|---|
| Cases with table-cell items | 16 |
| Table-cell items per such case (median) | 7.5, from 4 physical rows |
| All table-cell items | 137, from 69 physical rows |
| Locator and id characters spent on cell items (median) | 1,068 of a 3,748-character prompt |
| Items of 12 characters or less per case (median) | 2. 15 cases have 6 or more. |

State of the 10 table cases after Prompts 1 to 4, using the scorer's rule that
the anchor must sit inside one item:

- **8 already have the anchor inside one evidence item.** They came through
  the prose layer or the xlsx cell text. These are PQA-004, 005, 007, 009,
  010, 011, 012 and 013. What is left for them is the model's choice, which a
  model run will show.
- **PQA-006 needs a wrapped line joined.** The source line is
  `Rs 1800/- per month Rs 2250/-`, and its last `per month` is on the next
  physical line.
- **PQA-008 needs geometry.** The text layer stores that row column by column
  (`Group `C'`, `employee`, `only`, `Rs 5125/ -`, ...). The model gets
  `Rs 4100/-` alone, and the revised `Rs 5125/ -` never reaches the prompt.

So the work that is still needed is narrower than "rows for 10 cases". Cell
text has to be verbatim, and a printed row has to be one band of lines.

## Proposal

### 1. Ingest: verbatim cell text (`ldi/pdftable.py`)

Read each cell's text from pdfium inside the cell box with
`textpage.get_text_bounded(left, bottom, right, top)`, instead of joining
glyphs. That is pdfium's own text for the box, with its spaces and
punctuation, so it is still verbatim source text.

Keep the glyph join only as a fallback, used when the bounded text is empty.

### 2. Ingest: a row band from geometry (`ldi/pdftable.py`, `ldi/spans.py`)

- **Ruled tables** (`p{n}r{k}`): a band is already the space between two
  horizontal rules. Join the lines inside a cell top to bottom with one space.
  This is close to what `_ruled_cells` does today, minus the spacing bug.
- **Text tables** (`p{n}t{k}`): start a new band at a horizontal rule when
  `_rules` finds rules that span the table. Without rules, start a new band
  when the vertical gap to the previous line is more than
  `TABLE_BAND_GAP x` the median line pitch inside the table.
- **No signal:** when neither signal is present, each printed line stays its
  own band. That is exactly today's behaviour.
- **XLSX:** the band is the sheet row. No change.
- **Storage:** keep cell `row` and `col` as they are. Add a `band` integer per
  table span, and store the cell box in the existing `bbox` column.
  `drafts_from_tables` passes `cell.bbox` and the band.

Span ids do not change shape. They stay `span_id(text, doc_sha8, occurrence)`
for each cell. Cells whose text gets fixed by step 1 get new ids, because the
id is a hash of the text. That is intended, and `EXTRACTOR_VERSION` is bumped
so old spans read as stale.

### 3. Evidence: one item per band (`ldi/evidence.py`)

In `_table_unit`, group the chosen cells by `(table_id, band)`, not by `line`.
Order the cells inside a band by visual line top to bottom, then by `col` left
to right. Return one source line per band:

- **Text:** the cell texts joined with one space, each cell verbatim. Never
  `\n`, because `eval/provenance._CITE` reads the quote from a single line.
  Never a separator such as `|` that the page does not print.
- **Ids:** every cell span id in that order, so each word in the quote traces
  back to one content-hashed span.

Header rows stay their own items, as today. Title rows too. A row item never
glues the header into its own text, because the header is not next to the row
on the page.

`Hit.source_span_ids` holds one id per source line. A row needs several, so
add `source_row_ids: tuple[tuple[str, ...], ...]` beside it, or allow a
space-separated id list in that slot. `_evidence_from_source` then treats a
band as one line for pinning, round-robin and dedupe. Its dedupe key is
`(path, first span id)`, so a cell already used inside a band is not quoted
again as a single cell.

### 4. Citation format for a row

The rendered citation stays as it is:

```
- cite: "Group `C' Group `C' (PCA) 05.03.2021 employee Rs 4100/- employee Rs 5125/ - only only"
  dae_allowances_2024.pdf, page 2, table p2t1 band 4, span 9a3d0c… 56e6e9… …
```

- **Quote:** the joined cell text, built in code and never typed by the model.
  `_collapse` folds whitespace on both sides, so one space between cells
  matches the page under the scorer and under `verify._cite`.
- **Locator:**
  `{path}, page {page}, table {table_id} band {band}, span {id1} {id2} …`.
  `_line_place` splits on `", span "`, so the place is the table band.
  `_sentence_around` then never merges a row with prose neighbours, and a row
  is already a complete unit.
- **Prompt view:** `format_evidence` prints the first id and the cell count,
  for example `span 9a3d0c…(+7)`. The full list stays on the `Evidence` item.
  That keeps the 1,068 characters of per-cell locators from coming back as
  one long line.

### 5. The release gate

- **`release_from_evidence`:** a row item is one item, so the model cites one
  id for the whole row. `_sentence_overlaps(..., slack=1)` sees the label
  words and the values together, which is the point of the change. Today a
  bare `Rs 4100/-` cell shares no word with the question and fails this gate.
- **`conflicting`:** a row with two rates has two integers, so `integer_values`
  is not a singleton and the row is skipped. That is right: existing and
  revised in one row are not a conflict. With one item per row, two separate
  single-number cells from the same row no longer appear as items, and can no
  longer be paired as a conflict.
- **`fallback_release`:** `_multi_number_grid` stays the guard. A row item
  without period headers does not parse as a grid. It is quoted as one
  sentence only when `_sentence_overlaps` passes in full, as for prose.
- **Abstention:** the exact not-found string is unchanged. A row that fails
  every gate still ends in `NOT_FOUND`.

### 6. `_parse_grid`

`_parse_grid` needs a run of period headers (`Q3 FY24 Q4 FY24 …`) before the
row tokens. A row item does not carry its header, so add an optional
`header: str = ""` field to `Evidence`. It is filled from the table's last
header row, joined the same way.

Then `_cell_from_question`, `_change_from_question` and `_placed_number` call
`_parse_grid(f"{item.header} {item.text}")` when `header` is set. The quoted
text still excludes the header.

Today's grid items are whole `finance_report_q3` style lines that already hold
their headers. Their `header` stays empty, so they parse exactly as now.

### 7. Evidence budget

- A band is one of the 24 `EVIDENCE_BUDGET` items. On the replay that is 137
  cell items down to 69 row items before bands, and fewer after bands, because
  a wrapped row counts once.
- In the 16 table cases this frees about 4 items each, for prose and other
  hits.
- **Character cap:** a band whose joined text is over `TABLE_ROW_CHARS`
  (proposed 400) falls back to its cells, so one wide row cannot take the
  prompt.
- **Pinned rows:** the cap stays `EVIDENCE_MATCHED_ROWS = 3` per table, now
  counted in bands.
- **Prompt characters:** expected to drop for table cases, because one locator
  replaces up to 7. Prose cases are unchanged.

## Rollout so nothing gets worse

1. **Behind a flag.** Put it behind `config.TABLE_ROW_EVIDENCE = False`. With
   the flag off, `_table_unit` and the locators are byte-identical to today.
   Step 1, the verbatim cell text, is an ingest change. It sits behind
   `EXTRACTOR_VERSION` and needs a re-ingest into a fresh replay copy, never
   into `ldi-eval-occ/idx`.
2. **Offline gate.** Use `/tmp/evidence_replay.py` on dev and held-out, plus
   the one-item anchor check. Turn the flag on only when no case loses its
   anchor and `facts_at_prompt` does not drop. Today that is 39 on dev and 19
   on held-out.
3. **Model gate.** Then run the user's model rescore of the 42 questions and
   the floor ids, after `ldi status`. Leave the flag off if any floor id
   regresses.

## Tests on synthetic tables

Fixtures are built in code: glyph boxes and rules passed straight to the
`pdftable` helpers, `DetectedCell` lists, pipe-table text, and
`Sheet!B3 value` cell lines. No PDF file or real document is needed, and
nothing is keyed to a case.

- `test_cell_text_keeps_spaces_and_periods`: a cell box containing
  `Rs 720/-` and `05.03.2021` reads back exactly that.
- `test_wrapped_label_is_one_band`: a 4-line label beside 2 rate cells on the
  last 2 lines becomes one band, ordered line by line, then column by column.
- `test_rules_split_bands`: two rows separated by a horizontal rule are two
  bands, even with a tight line pitch.
- `test_no_rule_and_even_pitch_keeps_lines`: with no signal, each printed
  line stays its own band, the same as today.
- `test_row_item_is_one_line_of_verbatim_cells`: the item text equals the
  space-joined cell texts, has no newline, and every cell span id appears in
  the locator in order.
- `test_row_span_ids_are_cell_hashes`: each id equals
  `span_id(cell.text, doc_sha8, occurrence)`, so no id is minted for the row.
- `test_two_row_header_and_merged_title`: the title and both header rows stay
  their own items, and `header` holds only the last header row.
- `test_empty_cells_do_not_add_spaces`: a band with empty middle cells joins
  with single spaces.
- `test_xlsx_merged_range_in_one_row`: a merged value copied across columns
  appears once per covered cell, as the sheet shows it, and the ids stay
  distinct by occurrence.
- `test_pipe_table_row`: a markdown pipe table row becomes one item without
  the `|` characters.
- `test_existing_and_revised_in_one_row_is_not_a_conflict`: `conflicting()`
  returns `None` for that row.
- `test_grid_header_field_parses_a_period_row`: `_parse_grid` on
  `header + text` finds the asked cell, and the citation quote excludes the
  header.
- `test_wide_row_falls_back_to_cells`: a band over `TABLE_ROW_CHARS` is
  quoted cell by cell.
- `test_flag_off_is_identical`: with `TABLE_ROW_EVIDENCE = False`,
  `build_evidence` output equals today's on the same fixtures.

## Risks

| Risk | Effect | Mitigation |
|---|---|---|
| pdfium bounded text differs from pypdf (for example `Rs 5125/ -`) | An anchor written from the pypdf layout may not match the cell | Both come from the same text layer. Measure the anchor hit rate on the replay copy before turning the flag on, and keep the prose spans as they are. |
| Wrong bands on skewed or rule-less PDFs | Two table rows merged into one item, which looks like a conflict or the wrong pairing | Bands only at rules or a clear gap. Otherwise one line per band, as today. Add a test with an even pitch. |
| Cells duplicated between band items and prose spans | The same value appears twice in the prompt | Deduplication is per span id. The prose layer is not touched, and pinning is shared. |
| Rows too wide for the budget | One row crowds out other hits | `TABLE_ROW_CHARS` cap with a fallback to cells, and round-robin across hits stays. |
| New span ids after the text fix | Saved citations to old ids go stale | Bump `EXTRACTOR_VERSION`. Ids remain content hashes. Re-ingest only into a replay copy. |
| Deterministic order | Reordered items change replay output | The order is fixed by `(table_id, band, line, col, span_id)`, and the flag-off test guards today's order. |
| Header context hides a mismatch | `_parse_grid` reads a header from the wrong table | The header comes only from the same `table_id`, and only its last header row. |
| Merged xlsx copies | A value is repeated in a row | That repeat matches what the sheet shows. The ids stay distinct by occurrence. |

## Out of scope

Dense retrieval, OCR tables, and any change to the scorer, `eval/floor_ids.json`
or the exact abstention string.
