
import json
from pypdf import PdfReader
import re

def sanitize_filename(s):
    # Replace any character that's not alphanumeric or underscore with an underscore
    return re.sub(r'[^A-Za-z0-9_]', '_', s)

def extract_single_chapter(reader, chapter_info):
    topic = chapter_info['topic']
    start = chapter_info['start']
    end = chapter_info['end']
    text = ""
    for page_num in range(start, end + 1):
        text += reader.pages[page_num].extract_text()
    return {"topic": topic, "text": text}

def write_chapter_to_txt(output, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Topic: {output['topic']}\n\n")
        f.write(output['text'])

def extract_selected_chapters(pdf_path, chapters_json, selected_chapter_keys):
    chapters = json.loads(chapters_json)
    reader = PdfReader(pdf_path)
    for chapter_key in selected_chapter_keys:
        chapter_info = chapters[chapter_key]
        output = extract_single_chapter(reader, chapter_info)
        safe_topic = sanitize_filename(chapter_info['topic'])
        filename = f"{chapter_key}_{safe_topic}.txt"

# Example usage:
chapters_json = '''
{
  "Chapter_1": {
    "topic": "cover",
    "start": 0,
    "end": 1
  },
  "Chapter_2": {
    "topic": "title page",
    "start": 2,
    "end": 3
  },
  "Chapter_3": {
    "topic": "about the authors",
    "start": 4,
    "end": 5
  },
  "Chapter_4": {
    "topic": "dedication",
    "start": 6,
    "end": 7
  },
  "Chapter_5": {
    "topic": "preface",
    "start": 8,
    "end": 18
  },
  "Chapter_6": {
    "topic": "acknowledgments for the global edition",
    "start": 19,
    "end": 21
  },
  "Chapter_7": {
    "topic": "table of contents",
    "start": 22,
    "end": 31
  },
  "Chapter_8": {
    "topic": "computer networks and the internet",
    "start": 32,
    "end": 111
  },
  "Chapter_9": {
    "topic": "application layer",
    "start": 112,
    "end": 211
  },
  "Chapter_10": {
    "topic": "transport layer",
    "start": 212,
    "end": 333
  },
  "Chapter_11": {
    "topic": "the network layer: data plane",
    "start": 334,
    "end": 407
  },
  "Chapter_12": {
    "topic": "the network layer: control plane",
    "start": 408,
    "end": 479
  },
  "Chapter_13": {
    "topic": "the link layer and lans",
    "start": 480,
    "end": 561
  },
  "Chapter_14": {
    "topic": "wireless and mobile networks",
    "start": 562,
    "end": 637
  },
  "Chapter_15": {
    "topic": "security in computer networks",
    "start": 638,
    "end": 721
  },
  "Chapter_16": {
    "topic": "references",
    "start": 722,
    "end": 761
  },
  "Chapter_17": {
    "topic": "index",
    "start": 762,
    "end": 796
  }
}
'''


if __name__ == "__main__":

  selected_chapters = ["Chapter_12"]

  # Get textbook name from db
  # Chapters json from pdf splitter
  # LLM select the chapters
  extract_selected_chapters(r"C:\Users\datex\Documents\Hackathon - 2025\HackPSUFA25\Sam\books\ComputerNetworking.pdf", chapters_json, selected_chapters)
