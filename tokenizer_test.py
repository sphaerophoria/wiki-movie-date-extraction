from pathlib import Path
from transformers import AutoTokenizer

model_name = "sentence-transformers/all-MiniLM-L6-v2"
text = "hello my name is mick"
tokenizer = AutoTokenizer.from_pretrained(model_name)
enc = tokenizer(
    text,
    truncation=True,
    max_length=512,
    return_offsets_mapping=True,
    return_tensors="pt",
)

for p in Path("inputs/preprocessed").iterdir():
    if (p.suffix == ".tokenized"):
        continue

    with open(p) as f:
        content = f.read()

    enc = tokenizer(content,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
        return_tensors="pt",
    )


    expected_attn_mask_sum = enc["attention_mask"].shape[0]
    for val in enc["attention_mask"].shape[1:]:
        expected_attn_mask_sum *= val

    assert(expected_attn_mask_sum == enc["attention_mask"].sum())

    assert(enc["input_ids"].shape[0] == 1)
    with open(str(p) + ".tokenized", "w") as f:
        for tok in enc["input_ids"][0].tolist():
            f.write(f"{tok} ")



#input_ids = enc["input_ids"]
#print(input_ids)

