import argparse
from pathlib import Path
from openai import OpenAI
import json

client = OpenAI(
    base_url="http://localhost:8082/v1",
    api_key="local",  # dummy for local servers
)

json_schema = {
  "type": "array",
  "description": "",
  "uniqueItems": True,
  "items": {
    "type": "object",
    "required": [
      "date",
      "medium"
    ],
    "properties": {
      "date": {
        "type": ["string", "null"],
      },
      "medium": {
          "type": "object",
          "required": [
            "home", "theatrical"
          ],
          "properties": {
              "home": {
                  "type": "boolean"
               },
              "theatrical": {
                  "type": "boolean"
               }
          }
      }
    }
  }
}

def llmExtractDates(f_content):

    resp = client.chat.completions.create(
        model="local-model-name",
        messages=[
            {"role": "system", "content": "Return output that matches the given JSON schema."},
            {"role": "user", "content": "In the following text extract any movie release dates (including home video releases) for movie the article is about (denoted by the first line). Please give the date verbatim as it was in the input. Approximate dates are ok (month + year, or year only). Please classify release medium as \"home\" or \"theatrical\". Home releases are when the movie was released for viewing on a home screen (dvd, blu ray, vhs, streaming services are ok, but soundtrack release are not). Some dates will be related to other films (sequels, etc.). Please do not include these in the output. If there is no release date found, please return an empty array"},
            {"role": "user", "content": "<file>" + f_content + "</file>"},
        ],
        temperature=0.2,
        response_format={
            "type": "json_object",
            "schema": json_schema,
        },
    )

    content = resp.choices[0].message.content
    data = json.loads(content)
    return data

def llmExtractDatesMock(f_content):
    return [{'date': 'March 23, 2010', 'medium': {'home': True, 'theatrical': False}}, {'date': 'November 1, 2011', 'medium': {'home': True, 'theatrical': False}}, {'date': 'June 4, 2019', 'medium': {'home': True, 'theatrical': False}}]

def llmExtractLabels(file_name):
    print(f"Looking at {file_name}")
    with open(file_name) as f:
        f_content = f.read()

    llm_json_path = Path(str(file_name) + ".llm.json")
    if llm_json_path.exists():
        print(f"  Re-parsing cached LLM output")
        with open(llm_json_path) as f:
            extracted = json.loads(f.read())
    else:
        extracted = llmExtractDates(f_content)
        with open(llm_json_path, "w") as f:
            f.write(json.dumps(extracted))

    #extracted = llmExtractDatesMock(f_content)
    labels = []
    for item in extracted:
        if item["date"] is None:
            continue

        start = f_content.index(item["date"])
        print(item["date"])
        print(len(item["date"]))
        end = start + len(item["date"])

        label = {
            "date_start": start,
            "date_end": end,
            "medium": item["medium"]
        }
        labels.append(label)
    return labels


parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help="Directory containing preprocessed text files")
args = parser.parse_args()

sample_dir = Path(args.data_dir)

paths = [p for p in sample_dir.iterdir() if p.suffix != ".json"]
total = len(paths)

for i, path in enumerate(sorted(paths), 1):
    print(f"[{i}/{total}]", end=" ")
    json_path = Path(str(path) + ".json")
    if json_path.exists():
        print(f"Skipping {path} (already done)")
        continue
    try:
        labels = llmExtractLabels(path)
    except Exception as e:
        print(f"Failed to process {path} ({e})")
        continue

    with open(json_path, "w") as f:
        f.write(json.dumps(labels))


#print(llmExtractLabels("/home/streamer/work/date-extraction-test/preprocessed/Toy_Story_2_32"))
