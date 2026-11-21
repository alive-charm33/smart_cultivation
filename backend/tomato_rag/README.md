# Tomato disease + growth stage knowledge base

- `docs/diseases/`  10 files, one per classifier class (filter by `disease_id`)
- `docs/stages/`    7 growth-stage files (filter by `stage_id`)
- `docs/cross/`     leaf-spot comparison and disease x stage risk matrix
- `label_map.json`  classifier label -> disease_id

## Linking diseases and stages
- Each disease file has `risk_stages` (front matter) and a "Stage-wise action" table.
- Each stage file has `diseases_at_risk` (front matter) and a risk table.
- IDs used everywhere: nursery, transplanting, vegetative, flowering_fruit_set,
  fruit_development, ripening_harvest, end_of_season.

## Before using in production
1. Verify every chemical against CIB&RC (cibrc.nic.in) for tomato in India and your state's recommendation (MPKV Rahuri / Maharashtra agriculture department).
2. Add doses only from the product label or state package of practices; none are included on purpose.
3. Add the exact source URL and access date to each file's front matter after you verify against the source pages.
4. Have an agronomist or KVK expert review the files.

## Pipeline
```
pip install -r requirements.txt
python build_index.py                       # chunk by section + index in Chroma
python eval_retrieval.py                    # test retrieval alone
export GROQ_API_KEY=...                     # LLM step (Groq). Windows PowerShell: $env:GROQ_API_KEY="..."
python rag.py --image leaf.jpg --model efficientnetb0_tomato.keras \
              --stage flowering_fruit_set --question "How do I treat this?"
python rag.py --simulate "Tomato___Early_blight=0.55,Tomato___Target_Spot=0.35,Tomato___Septoria_leaf_spot=0.10" --no-llm
```
Set `EMBED_MODEL=intfloat/multilingual-e5-base` (and rebuild the index) for Marathi/Hindi questions.
Make sure `class_names.json` is in the same order as your training generator's class indices.

## Growth stage by days
- Day ranges live in each `docs/stages/*.md` front matter (`day_from`, `day_to`, `day_reference`) and in its `## Timing` section.
- `stages.py` reads them and converts days to stage(s); overlapping stages are all returned.
- CLI: `--days 47` (days since transplanting), `--days 40 --days-from sowing --nursery-days 25`, or `--stage flowering_fruit_set`.
- If `--stage` (what the farmer sees on the plant) and `--days` are both given, the stage wins.
- Verify the ranges against ICAR-IIHR / TNAU / your seed company's guide for your hybrid.

## LLM provider
Default is Groq (`GROQ_API_KEY`, model `llama-3.3-70b-versatile`). Change with `LLM_MODEL=...`; check console.groq.com/docs/models for current names. Set `LLM_PROVIDER=anthropic` to use Claude instead.
