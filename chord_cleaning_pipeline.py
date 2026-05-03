"""
chord_cleaning_pipeline.py
==========================
Automated LLM-powered chord content cleaning pipeline with MULTI-PROVIDER
failover support (Gemini ↔ Groq) and resume capability.

Reads a raw CSV of scraped Indonesian pop song chords, sends each row
(one song = one chunk) to an LLM API with a system prompt for cleaning,
then reconstructs the cleaned CSV incrementally.

Author  : Pipeline Generator
Date    : 2026-03-12
Python  : >= 3.10
Deps    : pandas, google-genai, groq
"""

import csv
import io
import os
import re
import sys
import time
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

# ──────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────

# ── Provider selection ───────────────────────────────────────────────
# Options: "gemini", "groq"
# The pipeline tries the selected provider first. If all retries fail,
# it automatically falls back to the other provider for that row.
ACTIVE_PROVIDER: str = "groq"  # ← change to "gemini" when quota resets

# ── Gemini settings ──────────────────────────────────────────────────
GEMINI_API_KEY: str = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or "AIzaSyBjsdYS9QAyLXuFv1UJE7ZSbe6F3ID-4ps"
)
GEMINI_MODEL: str = "gemini-2.5-flash"

# ── Groq settings (free tier: 30 req/min, 6000 tokens/min) ──────────
GROQ_API_KEY: str = (
    os.getenv("GROQ_API_KEY")
    or "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"  # ← paste your free key from https://console.groq.com
)
GROQ_MODEL: str = "llama-3.3-70b-versatile"  # free, strong, 128k context

# ── File paths ───────────────────────────────────────────────────────
INPUT_CSV: Path = Path(__file__).parent / "input_scrapping_result.csv"
OUTPUT_CSV: Path = Path(__file__).parent / "cleaned_dataset.csv"
FAILED_LOG: Path = Path(__file__).parent / "failed_rows.log"

# ── Rate-limit settings ─────────────────────────────────────────────
SLEEP_BETWEEN_CALLS: int = 12  # seconds between API calls
MAX_RETRIES: int = 4  # retries per row per provider
RETRY_BACKOFF_BASE: int = 20  # base seconds for exponential backoff

# ── Resume mode ──────────────────────────────────────────────────────
# If True, the pipeline checks how many rows are already in the output
# CSV and skips them — useful after a crash or quota exhaustion.
RESUME_MODE: bool = True

# ──────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT — your cleaning instructions
# ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT: str = """\
Bertindaklah sebagai Senior AI Engineer dan Data Annotator ahli di bidang Music Information Retrieval (MIR).

[INSIGHT]
Saya sedang menyusun dataset sekuensial kord lagu pop Indonesia untuk melatih model Deep Learning (LSTM) yang bertujuan memprediksi fungsi harmoni. Data mentah ini di-scrape dari situs penyedia kord (ChordTela) dalam format CSV. Namun, kolom `chord_content`-nya masih berantakan: penanda bagian lagu tidak konsisten, menyatu dengan kord, dan terdapat instruksi repetisi lisan (seperti "Kembali ke Reff" atau "Ulangi *") yang tidak bisa dibaca oleh algoritma pemecah baris.

[STATEMENT]
Tugasmu adalah membersihkan dan memformat ulang isi dari kolom `chord_content` di dalam data CSV yang saya berikan. Terapkan aturan pembersihan (CLEAR Rules) berikut secara mutlak:
1. Standardisasi Tag: Ubah semua penanda bagian lagu menjadi format kurung siku yang baku, yaitu: [Intro], [Verse], [Pre-Chorus], [Chorus], [Interlude], [Bridge], dan [Outro]. (Ubah teks seperti "Reff:" menjadi "[Chorus]", atau "*)" menjadi "[Pre-Chorus]", atau "Verse:" menjadi "[Verse]", atau "Musik :" menjadi "[Interlude]").
2. Isolasi Tag: Pastikan setiap tag (misal: [Intro]) berdiri sendiri di satu baris. Jika ada kord yang sebaris dengan tag, turunkan kord tersebut ke baris baru di bawahnya.
3. Ekspansi Repetisi Mutlak: HAPUS SEMUA kalimat instruksi seperti "Kembali ke : Reff", "Ulangi *", atau "Back to Verse". Ganti kalimat tersebut dengan melakukan COPY-PASTE (menyalin ulang) lirik dan kord secara utuh dari bagian target yang dirujuk.
4. Pertahankan Teks Asli: Jangan menghapus lirik lagu dan jangan mengubah format spasi antar kord. Biarkan lirik dan kord berjejer seperti aslinya.
5. Integritas CSV: Kembalikan data dalam bentuk CSV utuh dengan header kolom yang persis sama dengan input awal (title, artist, url, ..., harmonic_map).
6. Ekspansi Repetisi Relatif: Jika ada instruksi seperti "Ulangi 2x" atau "Ulangi 3x", lakukan ekspansi dengan menyalin ulang bagian yang dimaksud sebanyak jumlah yang diperintahkan.

[PERSONALITY]
Jadilah sangat teliti, presisi, dan kaku terhadap aturan format. Jangan tambahkan teks penjelasan, pengantar, atau penutup.

[EXPERIMENT]
Berikan hasil akhir murni hanya berupa data CSV di dalam satu "code block" agar bisa langsung saya salin dan gunakan di pipeline saya.

Berikut adalah data CSV mentah yang harus kamu bersihkan:
"""

# ──────────────────────────────────────────────────────────────────────
#  LOGGING
# ──────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("chord_pipeline")


# ──────────────────────────────────────────────────────────────────────
#  LLM PROVIDER ABSTRACTION
# ──────────────────────────────────────────────────────────────────────


def call_gemini(prompt: str) -> str:
    """Call Google Gemini API via the new google-genai SDK.

    Returns
    -------
    str
        Raw text response.
    """
    from google import genai as google_genai

    client = google_genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text


def call_groq(prompt: str) -> str:
    """Call Groq API (free tier, very fast inference on open models).

    Returns
    -------
    str
        Raw text response.
    """
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=8192,
    )
    return response.choices[0].message.content


def call_llm(prompt: str, provider: str) -> str:
    """Unified LLM dispatch.

    Parameters
    ----------
    prompt : str
        Full prompt text.
    provider : str
        "gemini" or "groq".

    Returns
    -------
    str
        Raw LLM text response.
    """
    if provider == "gemini":
        return call_gemini(prompt)
    elif provider == "groq":
        return call_groq(prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_fallback_provider(primary: str) -> str:
    """Return the other provider name for failover."""
    return "groq" if primary == "gemini" else "gemini"


def parse_retry_delay(error_msg: str) -> Optional[float]:
    """Extract suggested retry delay from an API error message.

    Looks for patterns like ``retryDelay: '39s'`` or ``retry in 39.8s``.

    Returns
    -------
    Optional[float]
        Seconds to wait, or None if not found.
    """
    match = re.search(
        r"retry\s*(?:in|Delay['\"]?\s*[:=]\s*['\"]?)\s*([\d.]+)\s*s",
        error_msg,
        re.IGNORECASE,
    )
    if match:
        return float(match.group(1))
    return None


# ──────────────────────────────────────────────────────────────────────
#  CSV HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────


def load_input_csv(path: Path) -> pd.DataFrame:
    """Read the raw scraping CSV into a DataFrame (all columns as str)."""
    logger.info("Loading input CSV: %s", path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    logger.info("Loaded %d rows × %d columns.", len(df), len(df.columns))
    return df


def count_existing_output_rows(path: Path) -> int:
    """Count data rows already in output CSV (for resume mode).

    Returns 0 if the file doesn't exist or is empty/corrupt.
    """
    if not path.exists():
        return 0
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        return len(df)
    except Exception:
        return 0


def row_to_csv_text(header: str, row: pd.Series) -> str:
    """Serialize one DataFrame row into a two-line CSV string (header + data)."""
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_ALL)
    writer.writerow(row.values)
    return f"{header}\n{buf.getvalue().strip()}"


def build_prompt(system_prompt: str, csv_chunk: str) -> str:
    """Concatenate system prompt with the CSV chunk."""
    return f"{system_prompt}\n{csv_chunk}"


def strip_markdown_codeblock(text: str) -> str:
    """Remove markdown code fences wrapping the LLM output."""
    text = re.sub(r"^```(?:csv|CSV)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text.strip())
    return text.strip()


def extract_data_row(llm_output: str, expected_columns: int) -> Optional[str]:
    """Parse LLM response to extract the cleaned data row (skip header).

    Returns
    -------
    Optional[str]
        Re-serialized CSV data line, or None if parsing fails.
    """
    cleaned = strip_markdown_codeblock(llm_output)
    lines = cleaned.strip().splitlines()

    if len(lines) < 2:
        logger.warning("LLM returned fewer than 2 lines; attempting single-line parse.")
        if len(lines) == 1:
            return lines[0]
        return None

    header_line = lines[0]
    data_text = "\n".join(lines[1:])

    try:
        reader = csv.reader(io.StringIO(f"{header_line}\n{data_text}"))
        _ = next(reader)  # skip header
        data_row = next(reader)
        if len(data_row) != expected_columns:
            logger.warning(
                "Column count mismatch: expected %d, got %d.",
                expected_columns,
                len(data_row),
            )
        buf = io.StringIO()
        writer = csv.writer(buf, quoting=csv.QUOTE_ALL)
        writer.writerow(data_row)
        return buf.getvalue().strip()
    except (csv.Error, StopIteration) as exc:
        logger.error("CSV parsing failed: %s", exc)
        return None


def init_output_csv(path: Path, header: str) -> None:
    """Write the CSV header to the output file (overwrite)."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(header + "\n")
    logger.info("Initialized output CSV: %s", path)


def append_row_to_csv(path: Path, row_text: str) -> None:
    """Append a single cleaned data row to the output CSV."""
    with open(path, "a", encoding="utf-8", newline="") as f:
        f.write(row_text + "\n")


def log_failure(path: Path, index: int, title: str, error: str) -> None:
    """Log a failed row for manual review."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[Row {index}] {title} — {error}\n")


# ──────────────────────────────────────────────────────────────────────
#  LLM CALL WITH RETRY + FAILOVER
# ──────────────────────────────────────────────────────────────────────


def call_llm_with_failover(prompt: str, primary: str) -> Optional[str]:
    """Try the primary provider with retries, then the fallback provider.

    Parameters
    ----------
    prompt : str
        Full prompt.
    primary : str
        Primary provider name ("gemini" or "groq").

    Returns
    -------
    Optional[str]
        LLM response text, or None if both providers fail.
    """
    fallback = get_fallback_provider(primary)

    for provider in [primary, fallback]:
        logger.info("  🔌 Trying provider: %s", provider.upper())
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info("    Attempt %d/%d ...", attempt, MAX_RETRIES)
                result = call_llm(prompt, provider)
                return result  # success

            except Exception as exc:
                last_error = str(exc)
                is_rate_limit = any(
                    kw in last_error.lower()
                    for kw in ["429", "resource_exhausted", "rate_limit", "quota"]
                )

                if is_rate_limit:
                    suggested = parse_retry_delay(last_error)
                    wait = suggested if suggested else RETRY_BACKOFF_BASE * attempt
                    wait = min(wait, 120)  # cap at 2 minutes
                    logger.warning(
                        "    ⚠ Rate-limited. Waiting %.0fs (attempt %d/%d) ...",
                        wait,
                        attempt,
                        MAX_RETRIES,
                    )
                    time.sleep(wait)
                else:
                    logger.error("    ✗ Error: %s", last_error[:200])
                    if attempt < MAX_RETRIES:
                        time.sleep(SLEEP_BETWEEN_CALLS)

        logger.warning(
            "  Provider %s exhausted after %d attempts. Trying fallback ...",
            provider.upper(),
            MAX_RETRIES,
        )

    logger.error("  ✗ ALL PROVIDERS FAILED.")
    return None


# ──────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────


def run_pipeline() -> None:
    """Execute the full cleaning pipeline:
    read → chunk (1 row) → inject prompt → call LLM → reconstruct CSV.
    """
    # ── 0. Validate API keys ─────────────────────────────────────────
    provider = ACTIVE_PROVIDER.lower()

    if provider == "gemini" and (not GEMINI_API_KEY or "YOUR_" in GEMINI_API_KEY):
        logger.error("Gemini API key not set.")
        sys.exit(1)
    if provider == "groq" and (not GROQ_API_KEY or "YOUR_" in GROQ_API_KEY):
        logger.error(
            "Groq API key not set. Get one FREE at https://console.groq.com "
            "then set GROQ_API_KEY env var or edit the script."
        )
        sys.exit(1)

    model_name = GEMINI_MODEL if provider == "gemini" else GROQ_MODEL
    logger.info("Primary provider: %s | Model: %s", provider.upper(), model_name)
    logger.info("Fallback provider: %s", get_fallback_provider(provider).upper())

    # ── 1. Load data ─────────────────────────────────────────────────
    df = load_input_csv(INPUT_CSV)
    columns = list(df.columns)
    num_columns = len(columns)

    # Build CSV header
    hdr_buf = io.StringIO()
    hdr_writer = csv.writer(hdr_buf, quoting=csv.QUOTE_ALL)
    hdr_writer.writerow(columns)
    csv_header = hdr_buf.getvalue().strip()

    # ── 2. Resume logic ──────────────────────────────────────────────
    skip_count = 0
    if RESUME_MODE and OUTPUT_CSV.exists():
        skip_count = count_existing_output_rows(OUTPUT_CSV)
        if skip_count > 0:
            logger.info(
                "🔄 RESUME MODE: %d rows already done. Skipping them.", skip_count
            )
        else:
            init_output_csv(OUTPUT_CSV, csv_header)
    else:
        init_output_csv(OUTPUT_CSV, csv_header)

    # Clear failure log only on fresh start
    if skip_count == 0 and FAILED_LOG.exists():
        FAILED_LOG.unlink()

    total = len(df)
    success_count = skip_count
    fail_count = 0

    # ── 3. Process row-by-row (1 song = 1 chunk) ────────────────────
    for idx, row in df.iterrows():
        if idx < skip_count:
            continue

        song_title = row.get("title", f"Row_{idx}")
        logger.info("━━━ [%d/%d] Processing: %s ━━━", idx + 1, total, song_title)

        # 3a. Serialize to CSV text
        csv_chunk = row_to_csv_text(csv_header, row)

        # 3b. Build full prompt
        full_prompt = build_prompt(SYSTEM_PROMPT, csv_chunk)

        # 3c. Call LLM with failover
        llm_output = call_llm_with_failover(full_prompt, provider)

        # 3d. Handle total failure
        if llm_output is None:
            fail_count += 1
            logger.error("  ✗ FAILED: %s", song_title)
            log_failure(FAILED_LOG, idx, song_title, "All providers failed")

            # Fallback: write original uncleaned row
            original_buf = io.StringIO()
            original_writer = csv.writer(original_buf, quoting=csv.QUOTE_ALL)
            original_writer.writerow(row.values)
            append_row_to_csv(OUTPUT_CSV, original_buf.getvalue().strip())
            logger.info("  ↳ Original row written as fallback.")
        else:
            # 3e. Parse LLM output
            data_row = extract_data_row(llm_output, num_columns)

            if data_row is None:
                fail_count += 1
                logger.error("  ✗ PARSE ERROR: %s", song_title)
                log_failure(FAILED_LOG, idx, song_title, "LLM output unparseable")

                original_buf = io.StringIO()
                original_writer = csv.writer(original_buf, quoting=csv.QUOTE_ALL)
                original_writer.writerow(row.values)
                append_row_to_csv(OUTPUT_CSV, original_buf.getvalue().strip())
                logger.info("  ↳ Original row written as fallback.")
            else:
                append_row_to_csv(OUTPUT_CSV, data_row)
                success_count += 1
                logger.info("  ✓ Cleaned and appended.")

        # 3f. Rate-limit delay
        if idx < total - 1:
            logger.info("  ⏳ Sleeping %ds ...", SLEEP_BETWEEN_CALLS)
            time.sleep(SLEEP_BETWEEN_CALLS)

    # ── 4. Summary ───────────────────────────────────────────────────
    logger.info("═" * 55)
    logger.info("PIPELINE COMPLETE")
    logger.info("  Total rows : %d", total)
    logger.info("  Cleaned    : %d", success_count)
    logger.info("  Failed     : %d (see %s)", fail_count, FAILED_LOG.name)
    logger.info("  Output     : %s", OUTPUT_CSV)
    logger.info("═" * 55)


# ──────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()
