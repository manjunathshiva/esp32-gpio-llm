Put the four AI Studio replies here:

    indomain_1.txt   indomain_2.txt     <- data/eval_prompts/indomain.md,  run twice
    offdomain_1.txt  offdomain_2.txt    <- data/eval_prompts/offdomain.md, run twice

Then `uv run python data/ingest_eval.py`. The reply can be pasted raw --
a ```json fence or the {"response": "..."} wrapper is tolerated.

Keep these files. They are the source the eval sets are rebuilt from, and
regenerating them is not free.
