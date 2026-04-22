"""
DBLP XML Streaming Parser
==========================
Memory-efficient streaming parser for the DBLP XML dataset using lxml.iterparse.
Extracts key bibliographic fields and filters by year for subset creation.
"""

import gzip
from pathlib import Path
from typing import Optional

import pandas as pd
from lxml import etree


# DBLP record types we care about
RECORD_TYPES = {
    "article",         # Journal articles
    "inproceedings",   # Conference papers
    "proceedings",     # Conference proceedings volumes
    "book",            # Books
    "incollection",    # Book chapters
    "phdthesis",       # PhD theses
    "mastersthesis",   # Master's theses
}

# Mapping record types to venue categories
VENUE_CATEGORY = {
    "article": "journal",
    "inproceedings": "conference",
    "proceedings": "conference",
    "book": "other",
    "incollection": "other",
    "phdthesis": "other",
    "mastersthesis": "other",
}


def _get_text(element, tag: str) -> Optional[str]:
    """Extract text content from the first matching child element."""
    child = element.find(tag)
    if child is not None:
        # Handle nested markup (e.g., <title>Some <i>italic</i> text</title>)
        text = etree.tostring(child, method="text", encoding="unicode")
        return text.strip() if text else None
    return None


def _get_all_text(element, tag: str) -> list[str]:
    """Get text from ALL matching child elements (e.g., multiple <author> tags)."""
    results = []
    for child in element.findall(tag):
        text = etree.tostring(child, method="text", encoding="unicode")
        if text and text.strip():
            results.append(text.strip())
    return results


def parse_dblp_xml(
    xml_path: str | Path,
    min_year: int = 2010,
    max_records: Optional[int] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Stream-parse the DBLP XML file and extract bibliographic records.

    Parameters
    ----------
    xml_path : str or Path
        Path to dblp.xml or dblp.xml.gz
    min_year : int
        Minimum publication year to include (inclusive). Default: 2010
    max_records : int, optional
        Maximum number of records to parse (for testing). None = parse all.
    show_progress : bool
        Whether to show a tqdm progress bar.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: title, authors, year, venue, pub_type, venue_category, key
    """
    xml_path = Path(xml_path)

    records = []
    count = 0
    skipped = 0

    # Determine if gzipped
    if xml_path.suffix == ".gz":
        file_obj = gzip.open(xml_path, "rb")
    else:
        file_obj = open(xml_path, "rb")

    try:
        # Use iterparse for memory-efficient streaming
        # We need to handle DTD entities, so use recover=True
        context = etree.iterparse(
            file_obj,
            events=("end",),
            tag=list(RECORD_TYPES),
            load_dtd=True,
            recover=True,
            huge_tree=True,
        )

        for event, elem in context:
            # Extract year first for early filtering
            year_text = _get_text(elem, "year")
            if year_text:
                try:
                    year = int(year_text)
                except ValueError:
                    year = None
            else:
                year = None

            # Filter by year
            if year is not None and year < min_year:
                # Clean up memory
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
                skipped += 1
                continue

            # Extract fields
            title = _get_text(elem, "title")
            authors = _get_all_text(elem, "author")

            # Venue: journal for articles, booktitle for conference papers
            venue = _get_text(elem, "journal") or _get_text(elem, "booktitle")
            pub_type = elem.tag
            venue_category = VENUE_CATEGORY.get(pub_type, "other")

            # Get the DBLP key (unique identifier)
            key = elem.get("key", None)

            records.append({
                "title": title,
                "authors": authors,
                "year": year,
                "venue": venue,
                "pub_type": pub_type,
                "venue_category": venue_category,
                "key": key,
            })

            count += 1

            # Clean up processed element to free memory
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

            # Check max_records limit
            if max_records and count >= max_records:
                break

    finally:
        file_obj.close()

    print(f"\nParsed {count:,} records (skipped {skipped:,} records before {min_year})")

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Basic type cleanup
    if not df.empty:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["pub_type"] = df["pub_type"].astype("category")
        df["venue_category"] = df["venue_category"].astype("category")

    return df


def save_to_parquet(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save DataFrame to Parquet format for fast reuse."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert authors list to string for parquet compatibility
    df_save = df.copy()
    df_save["authors"] = df_save["authors"].apply(lambda x: "|||".join(x) if isinstance(x, list) else "")

    df_save.to_parquet(output_path, index=False, engine="pyarrow")
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved {len(df):,} records to {output_path} ({size_mb:.1f} MB)")


def load_from_parquet(parquet_path: str | Path) -> pd.DataFrame:
    """Load DataFrame from Parquet and reconstruct authors list."""
    df = pd.read_parquet(parquet_path)
    df["authors"] = df["authors"].apply(
        lambda x: x.split("|||") if isinstance(x, str) and x else []
    )
    return df


if __name__ == "__main__":
    # Quick test with a small subset
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/dblp.xml.gz"
    df = parse_dblp_xml(path, min_year=2020, max_records=1000)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
