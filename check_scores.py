import json

rows = [json.loads(l) for l in open('outputs/demo_mix/results.jsonl', encoding='utf-8')]
for r in sorted(rows, key=lambda x: -x['risk']):
    tag = 'SCAM' if r['label'] == 1 else ('genuine' if r['label'] == 0 else '?')
    print(f"{r['risk']:.2f}  {r['verdict']:<12}  {tag:<8}  {r['title']}")