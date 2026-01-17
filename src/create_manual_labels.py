"""
Script để tạo manual labels cho Lab 2 section 2.2.2

⚠️ YÊU CẦU QUAN TRỌNG từ text2.txt:
    "Manually label references for at least 5 publications"
    
Nghĩa là: Sinh viên PHẢI TỰ TAY LABEL, không được dùng automatic matching!

⚠️ YÊU CẦU BỔ SUNG (từ hướng dẫn Lab):
    - Mỗi publication MANUAL phải có ít nhất 20 VALID matches
    - BibTeX entry không có match = INVALID (không tính vào số matches)
    - Publication có < 20 valid matches = INVALID sample (không được tính)
    - Auto labels cho phép < 20 matches nhưng accuracy có thể thấp

Script này hỗ trợ:
1. Tìm publications có đủ potential matches (≥20) để label
2. Hiển thị BibTeX entries và candidates để sinh viên REVIEW
3. Sinh viên tự quyết định match nào đúng
4. Validate: chỉ save publication có ≥20 valid matches
5. Lưu kết quả vào manual_labels.json

Format output theo yêu cầu:
{
    "publication_id": {
        "bibtex_key": "arxiv_id",
        ...
    },
    ...
}
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Set recursion limit before any library imports
sys.setrecursionlimit(10000)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    # Disable pyparsing packrat to avoid recursion issues
    import pyparsing
    # For newer pyparsing versions, packrat is not enabled by default
    # Only disable if the method exists (older versions)
    if hasattr(pyparsing.ParserElement, 'disablePackrat'):
        pyparsing.ParserElement.disablePackrat()
except ImportError:
    pass

try:
    import bibtexparser
except ImportError:
    print("❌ bibtexparser not installed. Run: pip install bibtexparser")
    sys.exit(1)

try:
    from fuzzywuzzy import fuzz
except ImportError:
    print("❌ fuzzywuzzy not installed. Run: pip install fuzzywuzzy python-Levenshtein")
    sys.exit(1)


# ============================================================================
# CONSTANTS
# ============================================================================
MIN_VALID_MATCHES_MANUAL = 20  # Mỗi publication manual phải có >= 20 valid matches
MIN_PUBLICATIONS_MANUAL = 5    # Cần ít nhất 5 publications cho manual labels
MIN_SCORE_AUTO_ACCEPT = 85     # Score tối thiểu để auto-accept


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
def load_refs_bib(path):
    """Load refs.bib file và trả về dict {bib_key: entry}"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            bib_db = bibtexparser.load(f)
        return {entry['ID']: entry for entry in bib_db.entries}
    except Exception as e:
        return {}


def load_references_json(path):
    """Load references.json và trả về dict {arxiv_id: metadata}
    
    Chỉ trả về những entry có metadata thực sự (không rỗng)
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Filter: chỉ giữ entries có data thực sự
        # Use type().__name__ instead of isinstance() to avoid recursion issues
        valid_refs = {
            k: v for k, v in data.items() 
            if v and type(v).__name__ == 'dict' and len(v) > 0
        }
        return valid_refs
    except Exception as e:
        return {}


# ============================================================================
# CANDIDATE FINDING FUNCTIONS (adapted from 5paper_found.py)
# ============================================================================
def find_candidates_for_manual_labeling(output_dir, min_potential=20):
    """
    Quét thư mục OUTPUT và tìm papers có đủ potential matches để manual labeling.
    
    ⚠️ YÊU CẦU: Mỗi paper manual PHẢI có ít nhất 20 valid matches
    
    Tiêu chí:
    1. references.json PHẢI có dữ liệu (chứa các arXiv IDs để match)
    2. refs.bib PHẢI có entries (BibTeX entries từ paper)
    3. potential_matches = min(bib_count, arxiv_ref_count) >= min_potential
    
    Args:
        output_dir: Thư mục output chứa các publication
        min_potential: Số match tiềm năng tối thiểu (default=20 cho manual)
        
    Returns:
        list of candidate dicts sorted by potential (descending)
    """
    candidates = []
    
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"❌ Output directory không tồn tại: {output_dir}")
        return []
    
    # Lấy danh sách các thư mục con (mỗi thư mục là 1 paper)
    paper_dirs = [d for d in output_path.iterdir() if d.is_dir()]
    
    print(f"\n🔍 Đang quét {len(paper_dirs)} thư mục paper trong {output_dir}...")
    print("=" * 70)
    
    # Statistics
    stats = {
        'no_refs_json': 0,
        'empty_refs_json': 0,
        'no_refs_bib': 0,
        'empty_refs_bib': 0,
        'insufficient_potential': 0,
        'valid': 0
    }
    
    for paper_dir in paper_dirs:
        arxiv_id = paper_dir.name
        
        # 1. Kiểm tra references.json
        ref_json_path = paper_dir / 'references.json'
        if not ref_json_path.exists():
            stats['no_refs_json'] += 1
            continue
        
        refs_json_data = load_references_json(ref_json_path)
        if not refs_json_data:
            stats['empty_refs_json'] += 1
            continue
            
        # 2. Kiểm tra refs.bib
        refs_bib_path = paper_dir / 'refs.bib'
        if not refs_bib_path.exists():
            stats['no_refs_bib'] += 1
            continue
            
        bib_entries = load_refs_bib(refs_bib_path)
        if not bib_entries:
            stats['empty_refs_bib'] += 1
            continue
        
        # 3. Tính số cặp match tiềm năng
        num_bib_entries = len(bib_entries)
        num_arxiv_refs = len(refs_json_data)
        potential_matches = min(num_bib_entries, num_arxiv_refs)
        
        # 4. Check minimum requirement
        if potential_matches < min_potential:
            stats['insufficient_potential'] += 1
            continue
        
        stats['valid'] += 1
        candidates.append({
            'pub_id': arxiv_id,
            'bib_count': num_bib_entries,
            'arxiv_refs_count': num_arxiv_refs,
            'potential_matches': potential_matches,
            'path': str(paper_dir),
            'sample_arxiv_ids': list(refs_json_data.keys())[:5]  # Preview
        })

    # Thống kê
    print(f"\n📊 THỐNG KÊ QUÉT:")
    print(f"   - Tổng số paper: {len(paper_dirs)}")
    print(f"   - Không có references.json: {stats['no_refs_json']}")
    print(f"   - references.json rỗng: {stats['empty_refs_json']}")
    print(f"   - Không có refs.bib: {stats['no_refs_bib']}")
    print(f"   - refs.bib rỗng: {stats['empty_refs_bib']}")
    print(f"   - Potential < {min_potential}: {stats['insufficient_potential']}")
    print(f"   - ✅ Paper đủ điều kiện (potential ≥ {min_potential}): {stats['valid']}")
    
    if not candidates:
        print(f"\n❌ KHÔNG TÌM THẤY PAPER NÀO CÓ ≥ {min_potential} POTENTIAL MATCHES!")
        print("   💡 Gợi ý:")
        print("   1. Đã chạy scraping metadata để điền references.json chưa?")
        print("   2. Đã chạy parser để tạo refs.bib chưa?")
        print("   3. Thử giảm --min-matches nếu cần (cho auto labels)")
        return []

    # Sắp xếp theo potential matches (ưu tiên cao nhất)
    candidates.sort(key=lambda x: x['potential_matches'], reverse=True)
    
    return candidates


def display_top_candidates(candidates, num_display=10):
    """Hiển thị top candidates để chọn labeling"""
    print("\n" + "=" * 70)
    print(f"🏆 TOP {min(num_display, len(candidates))} PAPERS ĐỦ ĐIỀU KIỆN CHO MANUAL LABELING")
    print(f"   (Yêu cầu: mỗi paper phải có ≥ {MIN_VALID_MATCHES_MANUAL} valid matches)")
    print("=" * 70)
    
    for i, c in enumerate(candidates[:num_display], 1):
        print(f"\n{i:2d}. Paper: {c['pub_id']}")
        print(f"    📚 BibTeX entries: {c['bib_count']}")
        print(f"    🔗 arXiv references: {c['arxiv_refs_count']}")
        print(f"    ✨ Potential matches: {c['potential_matches']}")
    
    print("\n" + "=" * 70)


def analyze_single_paper(paper_dir):
    """Phân tích chi tiết một paper cụ thể"""
    paper_path = Path(paper_dir)
    
    print(f"\n🔬 PHÂN TÍCH CHI TIẾT: {paper_path.name}")
    print("=" * 70)
    
    refs_json = {}
    bib_entries = {}
    
    # Load references.json
    ref_json_path = paper_path / 'references.json'
    if ref_json_path.exists():
        refs_json = load_references_json(ref_json_path)
        print(f"\n📄 references.json: {len(refs_json)} valid arXiv references")
        if refs_json:
            print("   Sample arXiv IDs có trong references.json:")
            for arxiv_id, metadata in list(refs_json.items())[:10]:
                title = metadata.get('title', metadata.get('paper_title', 'N/A'))
                if title and len(title) > 50:
                    title = title[:50] + '...'
                print(f"   - {arxiv_id}: {title}")
            if len(refs_json) > 10:
                print(f"   ... và {len(refs_json) - 10} arXiv IDs khác")
    else:
        print("\n❌ Không tìm thấy references.json")
    
    # Load refs.bib
    refs_bib_path = paper_path / 'refs.bib'
    if refs_bib_path.exists():
        bib_entries = load_refs_bib(refs_bib_path)
        print(f"\n📄 refs.bib: {len(bib_entries)} BibTeX entries")
        if bib_entries:
            print("   Sample BibTeX keys:")
            for bib_key, entry in list(bib_entries.items())[:10]:
                title = entry.get('title', 'N/A')
                if title and len(title) > 40:
                    title = title[:40] + '...'
                print(f"   - {bib_key}: {title}")
            if len(bib_entries) > 10:
                print(f"   ... và {len(bib_entries) - 10} entries khác")
    else:
        print("\n❌ Không tìm thấy refs.bib")
    
    return refs_json, bib_entries


# ============================================================================
# MATCHING FUNCTIONS
# ============================================================================
def calculate_match_score(bib_entry, ref_data):
    """
    Tính điểm match giữa một BibTeX entry và một reference từ references.json
    
    Returns:
        tuple (score, details)
    """
    bib_title = bib_entry.get('title', '').lower().strip()
    ref_title = ref_data.get('paper_title', ref_data.get('title', '')).lower().strip()
    
    bib_authors = bib_entry.get('author', bib_entry.get('authors', '')).lower()
    ref_authors = ref_data.get('paper_authors', ref_data.get('authors', ''))
    
    # Use type().__name__ instead of isinstance() to avoid recursion issues
    if type(ref_authors).__name__ in ('list', 'tuple'):
        ref_authors = ' '.join(str(a) for a in ref_authors).lower()
    else:
        ref_authors = str(ref_authors).lower()
    
    # Title similarity (most important)
    title_score = fuzz.token_sort_ratio(bib_title, ref_title) if bib_title and ref_title else 0
    
    # Author overlap
    author_score = fuzz.token_set_ratio(bib_authors, ref_authors) if bib_authors and ref_authors else 0
    
    # Combined score
    combined_score = title_score * 0.7 + author_score * 0.3
    
    return combined_score, {
        'title_score': title_score,
        'author_score': author_score,
        'bib_title': bib_title[:60] + '...' if len(bib_title) > 60 else bib_title,
        'ref_title': ref_title[:60] + '...' if len(ref_title) > 60 else ref_title
    }


def find_best_matches(refs_bib, references, top_n=3):
    """
    Tìm best matches cho mỗi BibTeX entry
    
    Returns:
        dict of {bibtex_key: [(arxiv_id, score, details), ...]}
    """
    matches = {}
    
    for bib_key, bib_entry in refs_bib.items():
        candidates = []
        
        for arxiv_id, ref_data in references.items():
            if not ref_data:  # Skip empty entries
                continue
                
            score, details = calculate_match_score(bib_entry, ref_data)
            candidates.append((arxiv_id, score, details))
        
        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top N
        matches[bib_key] = candidates[:top_n]
    
    return matches


# ============================================================================
# INTERACTIVE LABELING
# ============================================================================
def interactive_labeling(output_dir, pub_id, min_matches=20, auto_mode=False):
    """
    Interactive MANUAL labeling cho một publication
    
    ⚠️ YÊU CẦU: 
    - Sinh viên phải TỰ TAY xác nhận từng match
    - Publication MANUAL phải có ≥ min_matches valid matches
    - BibTeX entry không match = INVALID (không tính)
    
    Args:
        output_dir: Thư mục output
        pub_id: Publication ID
        min_matches: Số valid matches tối thiểu để publication được tính
        auto_mode: False = manual confirm, True = auto-accept high scores
    
    Returns:
        tuple (labels_dict, stats_dict)
            - labels_dict: {bibtex_key: arxiv_id} cho các valid matches
            - stats_dict: thống kê về valid/invalid entries
    """
    pub_path = Path(output_dir) / pub_id
    refs_bib_path = pub_path / 'refs.bib'
    references_path = pub_path / 'references.json'
    
    if not refs_bib_path.exists():
        print(f"❌ Không tìm thấy refs.bib tại {refs_bib_path}")
        return {}, {'error': 'no_refs_bib'}
    
    if not references_path.exists():
        print(f"❌ Không tìm thấy references.json tại {references_path}")
        return {}, {'error': 'no_references_json'}
    
    refs_bib = load_refs_bib(refs_bib_path)
    references = load_references_json(references_path)
    
    print(f"\n{'='*70}")
    print(f"MANUAL LABELING PUBLICATION: {pub_id}")
    print(f"{'='*70}")
    print(f"📊 BibTeX entries: {len(refs_bib)}")
    print(f"📊 arXiv references (valid): {len(references)}")
    print(f"📊 Potential matches: {min(len(refs_bib), len(references))}")
    print(f"⚠️  Yêu cầu: ≥ {min_matches} valid matches để publication được tính")
    
    # Find best matches
    matches = find_best_matches(refs_bib, references)
    
    # Statistics
    stats = {
        'total_bib_entries': len(refs_bib),
        'total_arxiv_refs': len(references),
        'valid_matches': 0,
        'invalid_entries': 0,  # BibTeX entries không có match
        'skipped_entries': 0,   # User skipped manually
    }
    
    labels = {}
    
    if auto_mode:
        print("\n⚠️ WARNING: AUTO MODE - CHỈ ĐỂ TEST!")
        print("Yêu cầu thật sự: Phải MANUAL REVIEW từng pair")
    
    print("\n" + "-"*70)
    print("HƯỚNG DẪN:")
    print("- Hệ thống hiển thị BibTeX entry và top 3 candidates")
    print("- BẠN xem xét và quyết định match nào đúng")
    print("- BibTeX entry KHÔNG CÓ MATCH sẽ được đánh dấu INVALID")
    print("- KHÔNG tính vào số valid matches")
    if not auto_mode:
        print("- Nhập số thứ tự (1-3) để chọn match")
        print("- Nhập 'n' để đánh dấu KHÔNG CÓ MATCH (invalid entry)")
        print("- Nhập 'q' để dừng labeling publication này")
    print("-"*70)
    
    processed = 0
    for bib_key, candidates in matches.items():
        processed += 1
        bib_entry = refs_bib[bib_key]
        bib_title = bib_entry.get('title', 'N/A')
        bib_author = bib_entry.get('author', 'N/A')
        
        print(f"\n{'─'*70}")
        print(f"[{processed}/{len(refs_bib)}] 📚 BibTeX Entry: {bib_key}")
        print(f"   Title: {bib_title}")
        print(f"   Author: {str(bib_author)[:100]}...")
        print(f"   ✅ Valid matches so far: {stats['valid_matches']} / {min_matches} required")
        
        # Check if any candidate exists
        valid_candidates = [c for c in candidates if c[1] > 0]  # Score > 0
        
        if not valid_candidates:
            print("   ⚠️ KHÔNG CÓ CANDIDATE NÀO!")
            print("   → Entry này sẽ được đánh dấu INVALID (không tính vào matches)")
            stats['invalid_entries'] += 1
            continue
        
        print(f"\n   Top {len(valid_candidates[:3])} candidates:")
        for i, (arxiv_id, score, details) in enumerate(valid_candidates[:3], 1):
            ref_data = references.get(arxiv_id, {})
            ref_title = ref_data.get('paper_title', ref_data.get('title', 'N/A'))
            ref_authors = ref_data.get('paper_authors', ref_data.get('authors', ''))
            if type(ref_authors).__name__ in ('list', 'tuple'):
                ref_authors = ', '.join(str(a) for a in ref_authors)
            
            print(f"\n   [{i}] arxiv_id: {arxiv_id}")
            print(f"       Title: {str(ref_title)[:80]}...")
            print(f"       Authors: {str(ref_authors)[:60]}...")
            print(f"       Score: {score:.1f} (title={details['title_score']}, author={details['author_score']})")
        
        # Decision making
        if auto_mode:
            # Auto mode: accept score >= MIN_SCORE_AUTO_ACCEPT
            best_arxiv_id, best_score, _ = valid_candidates[0]
            if best_score >= MIN_SCORE_AUTO_ACCEPT:
                labels[bib_key] = best_arxiv_id
                stats['valid_matches'] += 1
                print(f"\n   ✅ AUTO-SELECTED (score={best_score:.0f}): {best_arxiv_id}")
            else:
                stats['invalid_entries'] += 1
                print(f"   ❌ INVALID (score too low: {best_score:.0f} < {MIN_SCORE_AUTO_ACCEPT})")
        else:
            # MANUAL MODE
            while True:
                try:
                    choice = input(f"\n   👉 Chọn (1-{len(valid_candidates[:3])}) | 'n'=no match (invalid) | 'q'=quit: ").strip().lower()
                except EOFError:
                    # Non-interactive mode
                    stats['skipped_entries'] += 1
                    break
                
                if choice == 'q':
                    print("\n   ⏸️ Dừng labeling publication này")
                    # Return what we have
                    return labels, stats
                elif choice == 'n':
                    stats['invalid_entries'] += 1
                    print("   ❌ Đánh dấu INVALID - entry này không có match")
                    break
                elif choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(valid_candidates[:3]):
                        selected_id = valid_candidates[idx][0]
                        labels[bib_key] = selected_id
                        stats['valid_matches'] += 1
                        print(f"   ✅ Đã chọn: {selected_id}")
                        break
                    else:
                        print(f"   ❌ Số không hợp lệ, nhập 1-{len(valid_candidates[:3])}")
                else:
                    print("   ❌ Lựa chọn không hợp lệ")
    
    return labels, stats


# ============================================================================
# MAIN LABELING WORKFLOW
# ============================================================================
def create_manual_labels_interactive(output_dir, num_pubs=5, min_matches=20, 
                                     auto_mode=False, show_candidates=True):
    """
    Tạo manual labels cho publications
    
    ⚠️ YÊU CẦU MANUAL LABELS:
    - Mỗi publication PHẢI có ≥ min_matches valid matches
    - Publication có < min_matches valid matches = INVALID (không được tính)
    - Cần ít nhất num_pubs valid publications
    
    Args:
        output_dir: Thư mục output chứa các publication
        num_pubs: Số publication cần label (mặc định 5)
        min_matches: Số valid matches tối thiểu mỗi publication (mặc định 20)
        auto_mode: False = manual confirm, True = auto (CHỈ TEST)
        show_candidates: Hiển thị danh sách candidates trước khi bắt đầu
    
    Returns:
        dict: manual_labels theo format yêu cầu
    """
    # Step 1: Find candidates with enough potential
    candidates = find_candidates_for_manual_labeling(output_dir, min_potential=min_matches)
    
    if not candidates:
        return {}
    
    if show_candidates:
        display_top_candidates(candidates, num_display=15)
    
    # Check if we have enough candidates
    if len(candidates) < num_pubs:
        print(f"\n⚠️ CHỈ CÓ {len(candidates)} papers đủ điều kiện, cần {num_pubs}")
        print("   Tiếp tục với số papers hiện có...")
    
    if auto_mode:
        print("\n" + "="*70)
        print("⚠️⚠️⚠️ WARNING: AUTO MODE ENABLED ⚠️⚠️⚠️")
        print("Chế độ này CHỈ ĐỂ TEST - KHÔNG hợp lệ cho submission!")
        print("Yêu cầu thật sự: 'Manually label references'")
        print("="*70 + "\n")
    
    # Step 2: Interactive labeling
    manual_labels = {}
    valid_publications = 0
    invalid_publications = 0
    total_valid_matches = 0
    
    pub_index = 0
    while valid_publications < num_pubs and pub_index < len(candidates):
        candidate = candidates[pub_index]
        pub_id = candidate['pub_id']
        pub_index += 1
        
        print(f"\n{'='*70}")
        print(f"📖 Publication {valid_publications + 1}/{num_pubs}: {pub_id}")
        print(f"   (Candidate {pub_index}/{len(candidates)})")
        print(f"{'='*70}")
        
        labels, stats = interactive_labeling(output_dir, pub_id, 
                                              min_matches=min_matches,
                                              auto_mode=auto_mode)
        
        # Validate: publication phải có >= min_matches valid matches
        if labels and len(labels) >= min_matches:
            manual_labels[pub_id] = labels
            valid_publications += 1
            total_valid_matches += len(labels)
            print(f"\n✅ VALID PUBLICATION!")
            print(f"   Valid matches: {len(labels)} (≥ {min_matches} ✓)")
            print(f"   Invalid entries: {stats.get('invalid_entries', 0)}")
        else:
            invalid_publications += 1
            print(f"\n❌ INVALID PUBLICATION!")
            print(f"   Valid matches: {len(labels)} (< {min_matches} required)")
            print(f"   Invalid entries: {stats.get('invalid_entries', 0)}")
            print(f"   → Publication này KHÔNG được tính vào manual labels")
            
            if pub_index < len(candidates):
                print(f"   → Chuyển sang paper tiếp theo...")
            else:
                print(f"   ⚠️ Hết candidates để chọn!")
    
    # Step 3: Final summary
    print(f"\n{'='*70}")
    print("📊 FINAL SUMMARY")
    print("="*70)
    print(f"   Valid publications: {valid_publications}/{num_pubs} required")
    print(f"   Invalid publications (< {min_matches} matches): {invalid_publications}")
    print(f"   Total valid matches: {total_valid_matches}")
    print(f"   Average matches/pub: {total_valid_matches/valid_publications:.1f}" if valid_publications > 0 else "   Average: N/A")
    
    # Validate requirements
    print(f"\n{'─'*70}")
    print("📋 REQUIREMENTS CHECK:")
    
    req_pubs = valid_publications >= MIN_PUBLICATIONS_MANUAL
    req_matches = all(len(v) >= MIN_VALID_MATCHES_MANUAL for v in manual_labels.values())
    
    print(f"   [{'✅' if req_pubs else '❌'}] ≥ {MIN_PUBLICATIONS_MANUAL} valid publications: {valid_publications}")
    print(f"   [{'✅' if req_matches else '❌'}] Each pub has ≥ {MIN_VALID_MATCHES_MANUAL} valid matches")
    
    if req_pubs and req_matches:
        print(f"\n🎉 ALL REQUIREMENTS MET!")
    else:
        print(f"\n⚠️ REQUIREMENTS NOT MET")
        if not req_pubs:
            print(f"   Need {MIN_PUBLICATIONS_MANUAL - valid_publications} more valid publications")
    
    print("="*70)
    
    return manual_labels


def save_manual_labels(labels, output_path):
    """Save manual labels to JSON file
    
    Cũng in ra statistics về labels
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved manual labels to {output_path}")
    print(f"   Publications: {len(labels)}")
    
    total_pairs = sum(len(v) for v in labels.values())
    print(f"   Total pairs: {total_pairs}")
    
    # Per-publication breakdown
    print(f"\n   Per-publication breakdown:")
    for pub_id, matches in labels.items():
        status = "✅" if len(matches) >= MIN_VALID_MATCHES_MANUAL else "⚠️"
        print(f"   {status} {pub_id}: {len(matches)} matches")


def load_existing_labels(path):
    """Load existing manual labels để có thể tiếp tục labeling"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"⚠️ Error loading existing labels: {e}")
        return {}


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Create MANUAL labels for Lab 2 (Requirement 2.2.2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚠️ YÊU CẦU QUAN TRỌNG (từ hướng dẫn Lab):
    1. Mỗi publication MANUAL phải có ít nhất 20 VALID matches
    2. BibTeX entry không có match = INVALID (không tính vào số matches)
    3. Publication có < 20 valid matches = INVALID sample
    4. Cần ít nhất 5 valid publications
    
    --auto chỉ để TEST nhanh, KHÔNG được dùng cho submission!

Examples:
    # Manual labeling (recommended)
    python create_manual_labels.py --output-dir ../output
    
    # Scan only - không label, chỉ tìm candidates
    python create_manual_labels.py --output-dir ../output --scan-only
    
    # Auto mode for testing (NOT for submission)
    python create_manual_labels.py --output-dir ../output --auto
        """
    )
    parser.add_argument('--output-dir', default='../output', 
                       help='Output directory with publications')
    parser.add_argument('--num-pubs', type=int, default=5, 
                       help='Number of valid publications needed (default: 5)')
    parser.add_argument('--min-matches', type=int, default=20, 
                       help='Minimum valid matches per publication (default: 20)')
    parser.add_argument('--save-to', default='manual_labels.json', 
                       help='Output file path')
    parser.add_argument('--auto', action='store_true',
                       help='⚠️ AUTO MODE - CHỈ ĐỂ TEST!')
    parser.add_argument('--scan-only', action='store_true',
                       help='Chỉ scan và hiển thị candidates, không labeling')
    parser.add_argument('--analyze', type=str, default=None,
                       help='Analyze a specific publication (pub_id)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("📝 MANUAL LABELING TOOL - Lab 2 Section 2.2.2")
    print("="*70)
    print(f"Requirements:")
    print(f"   - ≥ {args.num_pubs} valid publications")
    print(f"   - Each publication: ≥ {args.min_matches} valid matches")
    print(f"   - BibTeX entries without match = INVALID")
    print("="*70)
    
    # Mode: Analyze single publication
    if args.analyze:
        pub_path = Path(args.output_dir) / args.analyze
        if pub_path.exists():
            analyze_single_paper(pub_path)
        else:
            print(f"❌ Publication không tồn tại: {args.analyze}")
        return
    
    # Mode: Scan only
    if args.scan_only:
        candidates = find_candidates_for_manual_labeling(
            args.output_dir, 
            min_potential=args.min_matches
        )
        if candidates:
            display_top_candidates(candidates, num_display=20)
        return
    
    # Mode: Auto labeling (TEST ONLY)
    if args.auto:
        print("\n⚠️⚠️⚠️ AUTO MODE - CHỈ ĐỂ TEST ⚠️⚠️⚠️")
        try:
            confirm = input("Bạn hiểu rằng auto mode KHÔNG hợp lệ cho submission? (yes/no): ")
            if confirm.lower() != 'yes':
                print("Đã hủy.")
                return
        except EOFError:
            pass  # Non-interactive mode, proceed
    
    # Mode: Manual labeling (default)
    labels = create_manual_labels_interactive(
        args.output_dir, 
        num_pubs=args.num_pubs,
        min_matches=args.min_matches,
        auto_mode=args.auto
    )
    
    # Save results
    if labels:
        save_manual_labels(labels, args.save_to)
        
        # Final validation
        valid_count = sum(1 for v in labels.values() if len(v) >= args.min_matches)
        if valid_count >= args.num_pubs:
            print(f"\n🎉 SUCCESS! Manual labels file is ready for evaluation")
        else:
            print(f"\n⚠️ WARNING: Only {valid_count}/{args.num_pubs} valid publications")
            print("   Consider adding more labels")
    else:
        print("\n❌ No valid labels created!")
        print("   Please try again with different publications")


if __name__ == "__main__":
    main()
