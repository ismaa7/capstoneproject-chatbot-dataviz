"""
Parses all Toyota PDF dossiers in data/dossiers/ and builds a ChromaDB vector store.
Run once: python -m src.ingest
"""
import os, re, hashlib
import fitz
import tiktoken
import chromadb
from chromadb.utils import embedding_functions
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL, COLLECTION_NAME, VECTORSTORE_DIR, PDF_DIR

TOYOTA_NAV_TABS = [
    "additional information", "specifications & equipment", "specifications",
    "equipment", "accessories", "wheels & trims", "trims & wheels", "colours",
    "grades", "connectivity & technology", "connectivity", "interior",
    "safety", "engines", "hybrid", "design",
]

SECTION_CANONICAL = {
    "additional information": "Owning",
    "specifications & equipment": "Specifications",
    "specifications": "Specifications",
    "equipment": "Equipment",
    "accessories": "Accessories",
    "wheels & trims": "Wheels & Trims",
    "trims & wheels": "Wheels & Trims",
    "colours": "Colours",
    "grades": "Grades",
    "connectivity & technology": "Connectivity",
    "connectivity": "Connectivity",
    "interior": "Interior",
    "safety": "Safety",
    "engines": "Hybrid & Engines",
    "hybrid": "Hybrid & Engines",
    "design": "Design",
}

@dataclass
class VehicleChunk:
    model: str
    section: str
    text: str
    source_file: str
    page_start: int
    page_end: int
    metadata: Dict = field(default_factory=dict)

_NAV_WORD_RE = re.compile(
    r'(?i)\b(design|hybrid|engines?|safety|interior|connectivity|grades?|'
    r'colours?|wheels?|trims?|accessories?|specifications?|equipment|'
    r'additional\s+information|owning|build\s+your|book\s+a|'
    r'find\s+a\s+dealer|keep\s+up|click\s+right)\b'
)

def strip_nav_bar(text: str) -> str:
    lines = text.splitlines()
    clean = []
    for line in lines:
        stripped = line.strip()
        if stripped and len(stripped) < 140:
            words = re.findall(r'[A-Za-z]+', stripped)
            if words:
                nav_hits = sum(1 for w in words if _NAV_WORD_RE.search(w))
                if nav_hits / len(words) > 0.60:
                    continue
        clean.append(line)
    return "\n".join(clean)

def detect_section_from_page(page_text: str) -> Optional[str]:
    clean = strip_nav_bar(page_text)
    first_600 = clean[:600].lower()
    for tab in TOYOTA_NAV_TABS:
        pattern = r'(?m)^\s*' + re.escape(tab) + r'\s*$'
        if re.search(pattern, first_600):
            return SECTION_CANONICAL[tab]
    return None

def detect_fuel_type(narrative_text: str) -> str:
    t = narrative_text.lower()
    if re.search(r'plug[-\s]in hybrid|\bphev\b|plug[-\s]in electric vehicle', t):
        return "Plug-in Hybrid"
    if re.search(r'\bbev\b|fully electric|pure electric|100%\s*electric|all[\s-]electric\s+vehicle|zero[\s-]emission vehicle', t):
        return "Electric"
    if re.search(r'\bbz\s*\d|\biz\s*\d|\be[-\s]tnga', t):
        return "Electric"
    if re.search(r'self[-\s]charging|petrol hybrid|\bhybrid\b|hybrid electric|\bhev\b', t):
        return "Hybrid"
    if re.search(r'\bhydrogen\b|\bfuel cell\b|\bfcev\b|\bmirai\b', t):
        return "Hydrogen"
    return "Petrol"

def detect_body_type(spec_text: str) -> str:
    t = spec_text
    if re.search(r'sports utility vehicle', t, re.I): return "SUV"
    if re.search(r'\d+-door hatchback', t, re.I):     return "Hatchback"
    if re.search(r'\bsaloon\b|\bsedan\b', t, re.I):  return "Saloon"
    if re.search(r'\bestate\b|\btourer\b', t, re.I): return "Estate"
    if re.search(r'\bmpv\b|\bpeople carrier\b', t, re.I): return "MPV"
    if re.search(r'\bcoupe\b|\bconvertible\b', t, re.I): return "Coupe"
    if re.search(r'\bpick[-\s]?up\b', t, re.I):      return "Pick-up"
    if re.search(r'\bsuv\b|\bcrossover\b', t, re.I): return "SUV"
    return "Unknown"

def extract_metadata_from_specs(spec_text: str) -> Dict:
    meta = {}
    din_hp_matches = re.findall(
        r'(\d{2,3})\s*(?:@\s*[\d,]+)?\s*DIN\s*hp|Total Hybrid System Output.*?(\d{3})\s*@',
        spec_text, re.I
    )
    powers = []
    for m in din_hp_matches:
        val = m[0] or m[1]
        if val and 50 <= int(val) <= 500:
            powers.append(int(val))
    if powers:
        meta["power_hp_min"] = min(powers)
        meta["power_hp_max"] = max(powers)
    disp = re.search(r'Displacement\s*\(cc\)[^\d]*(\d{3,4})', spec_text, re.I)
    if disp: meta["displacement_cc"] = int(disp.group(1))
    seats = re.search(r'Number\s+of\s+seats[^\d]*(\d)', spec_text, re.I)
    if seats: meta["seats"] = int(seats.group(1))
    co2 = re.search(r'CO.?2.*?(?:combined|WLTP).*?(\d{2,3})(?:[\-\u2013](\d{2,3}))?', spec_text, re.I)
    if co2:
        meta["co2_gkm_min"] = int(co2.group(1))
        meta["co2_gkm_max"] = int(co2.group(2)) if co2.group(2) else int(co2.group(1))
    boot = re.search(r'(?:5.seat mode|luggage capacity.*?5.seat).*?parcel shelf.*?(\d{3,4})', spec_text, re.I)
    if boot: meta["boot_litres"] = int(boot.group(1))
    else:
        b2 = re.search(r'luggage capacity[^\d]*(\d{3,4})', spec_text, re.I)
        if b2 and 100 <= int(b2.group(1)) <= 2000: meta["boot_litres"] = int(b2.group(1))
    meta["has_awd"] = bool(re.search(r'\bawd[-\s]?i?\b|all[-\s]wheel drive|4wd|four[-\s]wheel drive', spec_text, re.I))
    return meta

def extract_grades(grades_text: str) -> str:
    found = re.findall(
        r'\b(Icon|Design|Excel|GR\s*SPORT|Premiere\s*Edition|Active|Dynamic|'
        r'Style|Luna|Sol|Trek|Invincible|SR\+?|T-Spirit|T3|T4|T-Excel|'
        r'Adventure|Altitude)\b', grades_text
    )
    seen, unique = set(), []
    for g in found:
        g = re.sub(r'\s+', ' ', g).strip()
        if g not in seen:
            seen.add(g)
            unique.append(g)
    return ", ".join(unique)

def extract_model_name(filename: str) -> str:
    name = os.path.splitext(filename)[0]
    name = re.sub(r'(?i)toyota[-_ ]*', '', name)
    name = re.sub(r'[-_](\d{4}|UK|ES|EU|GR).*', '', name, flags=re.IGNORECASE)
    return name.replace('-', ' ').replace('_', ' ').strip().title()

def parse_toyota_pdf(pdf_path: str) -> List[VehicleChunk]:
    filename = os.path.basename(pdf_path)
    model = extract_model_name(filename)
    doc = fitz.open(pdf_path)
    page_texts = [page.get_text("text") for page in doc]
    doc.close()

    sections: List[Tuple[str, int, int, str]] = []
    current_section = "Overview"
    current_start = 1
    current_pages: List[str] = []

    for i, page_text in enumerate(page_texts):
        detected = detect_section_from_page(page_text)
        if detected and detected != current_section and current_pages:
            sections.append((current_section, current_start, i, " ".join(current_pages).strip()))
            current_section = detected
            current_start = i + 1
            current_pages = [page_text]
        else:
            current_pages.append(page_text)
    if current_pages:
        sections.append((current_section, current_start, len(page_texts), " ".join(current_pages).strip()))

    spec_text = hybrid_text = grades_text = ""
    for sn, _, _, st in sections:
        if "Specification" in sn or "Equipment" in sn: spec_text += st + " "
        if "Hybrid" in sn: hybrid_text += st + " "
        if "Grade" in sn: grades_text += st + " "

    narrative_for_fuel = hybrid_text if hybrid_text else " ".join(page_texts[:5])
    fuel_type = detect_fuel_type(narrative_for_fuel)
    body_type = detect_body_type(spec_text if spec_text else " ".join(page_texts))
    spec_meta = extract_metadata_from_specs(spec_text if spec_text else " ".join(page_texts))
    spec_meta.update({"fuel_type": fuel_type, "body_type": body_type, "model": model, "source_file": filename})
    if grades_text:
        g = extract_grades(grades_text)
        if g: spec_meta["grades"] = g

    chunks = []
    for sn, sp, ep, st in sections:
        if len(st.strip()) < 80: continue
        chunks.append(VehicleChunk(
            model=model, section=sn, text=st,
            source_file=filename, page_start=sp, page_end=ep,
            metadata={**spec_meta, "section": sn}
        ))

    print(f"  + {model} | Fuel: {fuel_type} | Body: {body_type} | Chunks: {len(chunks)}")
    return chunks

def split_large_chunk(chunk: VehicleChunk, max_tokens=7000) -> List[VehicleChunk]:
    enc = tiktoken.get_encoding('cl100k_base')
    if len(enc.encode(chunk.text)) <= max_tokens:
        return [chunk]
    sub_chunks, words, current_words, part = [], chunk.text.split(), [], 1
    for word in words:
        current_words.append(word)
        if len(enc.encode(' '.join(current_words))) >= max_tokens:
            new_meta = {**chunk.metadata, 'section': f"{chunk.section} (part {part})"}
            sub_chunks.append(VehicleChunk(
                model=chunk.model, section=f"{chunk.section} (part {part})",
                text=' '.join(current_words[:-1]), source_file=chunk.source_file,
                page_start=chunk.page_start, page_end=chunk.page_end, metadata=new_meta
            ))
            current_words = [word]
            part += 1
    if current_words:
        new_meta = {**chunk.metadata, 'section': f"{chunk.section} (part {part})"}
        sub_chunks.append(VehicleChunk(
            model=chunk.model, section=f"{chunk.section} (part {part})",
            text=' '.join(current_words), source_file=chunk.source_file,
            page_start=chunk.page_start, page_end=chunk.page_end, metadata=new_meta
        ))
    return sub_chunks

def chunk_to_id(chunk: VehicleChunk) -> str:
    return hashlib.md5(f"{chunk.source_file}_{chunk.section}_{chunk.page_start}".encode()).hexdigest()

def sanitise_metadata(meta: Dict) -> Dict:
    clean = {}
    for k, v in meta.items():
        if isinstance(v, list): clean[k] = ', '.join(str(i) for i in v)
        elif isinstance(v, (str, int, float, bool)): clean[k] = v
        else: clean[k] = str(v)
    return clean

def run_ingestion():
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDFs found in {PDF_DIR}")
        return
    print(f"Found {len(pdf_files)} PDF(s)\n")
    all_chunks = []
    for pdf_file in tqdm(pdf_files, desc="Parsing PDFs"):
        try:
            all_chunks.extend(parse_toyota_pdf(os.path.join(PDF_DIR, pdf_file)))
        except Exception as e:
            print(f"  ERROR {pdf_file}: {e}")

    enc = tiktoken.get_encoding('cl100k_base')
    fixed = []
    for c in all_chunks:
        fixed.extend(split_large_chunk(c) if len(enc.encode(c.text)) > 7000 else [c])
    all_chunks = fixed
    print(f"\n{len(all_chunks)} chunks ready to embed")

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name=EMBEDDING_MODEL)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=openai_ef, metadata={"hnsw:space": "cosine"}
    )

    existing_ids = set(collection.get()['ids'])
    added = skipped = 0
    for i in tqdm(range(0, len(all_chunks), 50), desc="Embedding & indexing"):
        batch = all_chunks[i:i+50]
        ids   = [chunk_to_id(c) for c in batch]
        texts = [c.text for c in batch]
        metas = [sanitise_metadata(c.metadata) for c in batch]
        new   = [(id_, t, m) for id_, t, m in zip(ids, texts, metas) if id_ not in existing_ids]
        if new:
            ids_n, texts_n, metas_n = zip(*new)
            collection.add(ids=list(ids_n), documents=list(texts_n), metadatas=list(metas_n))
            added += len(new)
        skipped += len(batch) - len(new)

    print(f"\nDone! Added: {added} | Skipped: {skipped} | Total in store: {collection.count()}")

if __name__ == "__main__":
    run_ingestion()
