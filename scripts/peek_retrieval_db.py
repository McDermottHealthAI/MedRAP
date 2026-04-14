"""Print the first 5 rows of the retrieval_db dataset."""

from datasets import load_from_disk

db = load_from_disk("data/retrieval_db")

for i, row in enumerate(db.select(range(5))):
    print(f"\n{'='*60}")
    print(f"Row {i}")
    print(f"{'='*60}")
    print(f"  id          : {row['id']}")
    print(f"  title       : {row['title']}")
    print(f"  doc_ids     : {row['doc_ids']}")
    print(f"  content     : {row['content'][:200]}...")
    print(f"  doc_text    : {row['doc_text'][:200]}...")
    print(f"  doc_tokens  : {row['doc_tokens'][:10]}...  (len={len(row['doc_tokens'])})")
    print(f"  doc_attn    : {row['doc_attention_mask'][:10]}...  (len={len(row['doc_attention_mask'])})")
    print(f"  embedding   : {row['doc_key_embeddings'][:4]}...  (dim={len(row['doc_key_embeddings'])})")
