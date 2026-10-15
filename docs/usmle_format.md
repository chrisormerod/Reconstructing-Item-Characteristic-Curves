# USMLE input format

Directory `data.usmle_dir`, one sheet per file (read with openpyxl; `~$*.xlsx` lock files are ignored).

| File | Rows | Columns used | Role |
|------|------|--------------|------|
| `train_final.xlsx` | 466 | `ItemNum, ItemStem_Text, Answer__A … Answer__J, Answer_Key, Answer_Text, ItemType, EXAM, Difficulty` | training + dev |
| `test_final.xlsx` | 201 | same without `Difficulty` | test content |
| `gold_final.xlsx` | 201 | `ItemNum, Difficulty` | test difficulties, joined on `ItemNum` |

`Response_Time` is never read. Parsing rules (`llm_icc/utils/usmle.py`):

- `ItemNum` is the unique `item_id`; the test and gold id sets must be identical (else an error).
- Options are the non-empty `Answer__X` cells; they must be contiguous from A, 4–10 per item.
- `Answer_Key` must be a letter pointing to a non-empty option. If `Answer_Text` disagrees with that option's
  text, a warning is logged and `Answer_Key` wins.
- `ItemType` (`Text`/`PIX`) and `EXAM` go to `item.metadata`. PIX stems refer to a missing picture; keep them
  (default) or drop them with `data.exclude_pix: true`.
- `Difficulty`: higher = harder. Stored in `reference_difficulty`; converted by `data.difficulty_transform`.
