import json
import re
from collections import defaultdict, Counter
from difflib import SequenceMatcher

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional imports
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    STOPWORDS = set(stopwords.words("english"))
except:
    STOPWORDS = set()

# -------------------- Setup --------------------
# Configure page with light theme
st.set_page_config(
    page_title="Plagiarism Analysis Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Plagiarism Analysis Tool - Compare student responses for similarity"
    }
)

@st.cache_resource(show_spinner=False)
def load_nlp():
    if not SPACY_AVAILABLE:
        return None
    try:
        return spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except:
        return None

NLP = load_nlp()

# -------------------- Text Extraction --------------------
def is_meaningful(value):
    """Return True only if value has real content (not null, empty, whitespace, or 'null')."""
    if value is None:
        return False
    s = str(value).strip()
    if s == "" or s.lower() == "null":
        return False
    return True

def extract_response_text(user_response):
    """Extract all meaningful text from userResponse, regardless of key names."""
    if not isinstance(user_response, dict):
        return ""
    parts = []
    for value in user_response.values():
        if is_meaningful(value):
            parts.append(str(value).strip())
    return " ".join(parts)

# -------------------- Text Processing --------------------
def normalize_text(text):
    """Lemmatize + remove stopwords (or just lowercase and filter if spacy unavailable)."""
    if not text:
        return ""
    text = text.lower()
    if NLP is not None:
        doc = NLP(text)
        return " ".join(t.lemma_ for t in doc if t.is_alpha and t.lemma_ not in STOPWORDS)
    else:
        # Fallback: simple word tokenization and stopword removal
        words = re.findall(r'[a-z]+', text)
        return " ".join(w for w in words if w not in STOPWORDS)

def char_ngram_set(text, n=3):
    """Generate character n-grams (case-insensitive, punctuation-normalized)."""
    # Lowercase and keep only alphanumeric + spaces
    text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text.lower())
    text = "_" + " ".join(text.split()) + "_"  # Normalize spaces
    return set(text[i:i+n] for i in range(len(text) - n + 1))

def char_ngram_jaccard(t1, t2, n=3):
    """Jaccard similarity on character n-grams."""
    s1, s2 = char_ngram_set(t1, n), char_ngram_set(t2, n)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

def word_jaccard(t1, t2):
    """Jaccard similarity on words."""
    s1, s2 = set(t1.split()), set(t2.split())
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

# -------------------- Highlighting --------------------
def highlight_common_sequences(text1, text2, min_length=5):
    """
    Highlight matching sequences - simple and direct approach.
    Uses SequenceMatcher on lowercase text, then highlights in original.
    """
    if not text1 or not text2:
        return text1, text2
    
    # Use lowercase for matching
    lower1 = text1.lower()
    lower2 = text2.lower()
    
    # Find matching blocks directly
    matcher = SequenceMatcher(None, lower1, lower2, autojunk=False)
    blocks = matcher.get_matching_blocks()
    
    spans1 = []
    spans2 = []
    
    for block in blocks:
        if block.size >= min_length:
            # Direct mapping - same positions in original text
            spans1.append((block.a, block.a + block.size))
            spans2.append((block.b, block.b + block.size))
    
    def apply_highlights(text, spans):
        if not spans:
            return text
        
        # Sort and merge overlapping/adjacent spans
        spans = sorted(spans)
        merged = []
        
        for start, end in spans:
            start = max(0, min(start, len(text)))
            end = max(start, min(end, len(text)))
            
            if start >= end:
                continue
            
            if merged and start <= merged[-1][1] + 2:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        
        # Apply highlighting
        result = ""
        last = 0
        for start, end in merged:
            result += text[last:start]
            result += f'<span style="background-color: #FFFF00; padding: 2px;">{text[start:end]}</span>'
            last = end
        result += text[last:]
        return result
    
    return apply_highlights(text1, spans1), apply_highlights(text2, spans2)

# -------------------- Main Analysis --------------------
def run_plagiarism_analysis(data, threshold, weights):
    """Group by (instanceId, activityId), compare different users, return results."""
    
    # Phase 1: Group responses
    groups = defaultdict(list)
    skipped = []
    
    for idx, entry in enumerate(data):
        instance_id = entry.get("instanceId")
        activity_id = entry.get("activityId")
        user_id = entry.get("userId")
        activity_type = entry.get("activityType", "Unknown")
        activity_name = entry.get("activityName", "Unknown")
        
        if not all([instance_id, activity_id, user_id]):
            skipped.append({"Index": idx, "Reason": "Missing instanceId/activityId/userId"})
            continue
        
        raw_text = extract_response_text(entry.get("userResponse", {}))
        if not raw_text.strip():
            skipped.append({"Index": idx, "Reason": "Empty response"})
            continue
        
        norm_text = normalize_text(raw_text)
        
        groups[(str(instance_id), str(activity_id))].append({
            "userId": str(user_id),
            "raw": raw_text,
            "norm": norm_text,
            "activityType": activity_type,
            "activityName": activity_name
        })
    
    # Phase 2: Compare pairs within each group
    detailed_rows = []
    
    for (instance_id, activity_id), responses in groups.items():
        if len(responses) < 2:
            continue
        
        users = [r["userId"] for r in responses]
        raw_texts = [r["raw"] for r in responses]
        norm_texts = [r["norm"] for r in responses]
        activity_name = responses[0]["activityName"]
        
        # TF-IDF similarity matrix (only if weight > 0)
        tfidf_sim = None
        if weights['tfidf'] > 0:
            try:
                vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
                tfidf_matrix = vectorizer.fit_transform(norm_texts)
                tfidf_sim = cosine_similarity(tfidf_matrix)
            except:
                tfidf_sim = [[0]*len(responses) for _ in range(len(responses))]
        
        # Compare all pairs (skip same user)
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                if users[i] == users[j]:
                    continue  # Skip self-comparisons
                
                # Calculate only if weight > 0
                tfidf_score = 0.0
                if weights['tfidf'] > 0 and tfidf_sim is not None:
                    tfidf_score = tfidf_sim[i][j] if hasattr(tfidf_sim, '__getitem__') else 0
                
                char3 = 0.0
                if weights['char3'] > 0:
                    char3 = char_ngram_jaccard(raw_texts[i], raw_texts[j], 3)
                
                char5 = 0.0
                if weights['char5'] > 0:
                    char5 = char_ngram_jaccard(raw_texts[i], raw_texts[j], 5)
                
                wj = 0.0
                if weights['word_jaccard'] > 0:
                    wj = word_jaccard(norm_texts[i], norm_texts[j])
                
                # Ensemble score using user-defined weights
                ensemble = (weights['tfidf'] * tfidf_score + 
                           weights['char3'] * char3 + 
                           weights['char5'] * char5 + 
                           weights['word_jaccard'] * wj)
                
                # Exact copy check
                exact_copy = raw_texts[i].strip().lower() == raw_texts[j].strip().lower()
                if exact_copy:
                    ensemble = 1.0
                
                detailed_rows.append({
                    "Instance ID": instance_id,
                    "Activity ID": activity_id,
                    "Activity Name": activity_name,
                    "User 1": users[i],
                    "User 2": users[j],
                    "TF-IDF (%)": round(tfidf_score * 100, 2),
                    "Char-3 (%)": round(char3 * 100, 2),
                    "Char-5 (%)": round(char5 * 100, 2),
                    "Word Jaccard (%)": round(wj * 100, 2),
                    "Exact Copy": "Yes" if exact_copy else "No",
                    "Ensemble (%)": round(ensemble * 100, 2),
                    "Answer 1": raw_texts[i],
                    "Answer 2": raw_texts[j]
                })
    
    return detailed_rows, skipped

# -------------------- Streamlit UI --------------------
# Force light theme with custom CSS
st.markdown("""
<style>
    /* Force light theme */
    .stApp {
        background-color: white;
        color: #262730;
    }
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
    /* Improve contrast */
    .stMarkdown, .stText, p {
        color: #262730;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Plagiarism Analysis Tool")

# Sidebar settings
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Flag Threshold (%)", 0, 100, 70, 5)
min_highlight_len = st.sidebar.slider("Min highlight length (chars)", 3, 15, 5, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Ensemble Weights")
st.sidebar.caption("Adjust weights for each similarity metric (must sum to 1.0)")

# Preset buttons
preset_col1, preset_col2 = st.sidebar.columns(2)
if preset_col1.button("Balanced", help="Equal weights for all metrics"):
    st.session_state['w_tfidf'] = 0.25
    st.session_state['w_char3'] = 0.25
    st.session_state['w_char5'] = 0.25
    st.session_state['w_word'] = 0.25
if preset_col2.button("TF-IDF Focus", help="Emphasize semantic similarity"):
    st.session_state['w_tfidf'] = 0.60
    st.session_state['w_char3'] = 0.15
    st.session_state['w_char5'] = 0.10
    st.session_state['w_word'] = 0.15

w_tfidf = st.sidebar.slider("TF-IDF Weight", 0.0, 1.0, 0.35, 0.05, key="w_tfidf")
w_char3 = st.sidebar.slider("Char-3 Jaccard Weight", 0.0, 1.0, 0.25, 0.05, key="w_char3")
w_char5 = st.sidebar.slider("Char-5 Jaccard Weight", 0.0, 1.0, 0.15, 0.05, key="w_char5")
w_word = st.sidebar.slider("Word Jaccard Weight", 0.0, 1.0, 0.25, 0.05, key="w_word")

# Calculate and display total with smart indicator
weights_sum = w_tfidf + w_char3 + w_char5 + w_word
weights = {
    'tfidf': w_tfidf,
    'char3': w_char3,
    'char5': w_char5,
    'word_jaccard': w_word
}

# Smart display based on sum
if abs(weights_sum - 1.0) < 0.001:
    st.sidebar.success(f"✓ Sum = {weights_sum:.2f}")
elif weights_sum == 0:
    st.sidebar.error(f"⚠️ All weights are 0! Set at least one weight > 0")
else:
    st.sidebar.info(f"ℹ️ Sum = {weights_sum:.2f} (will auto-normalize to 1.0)")
    
    # Add normalize button when sum != 1.0
    if st.sidebar.button("🔧 Normalize Weights Now"):
        if weights_sum > 0:
            st.session_state['w_tfidf'] = w_tfidf / weights_sum
            st.session_state['w_char3'] = w_char3 / weights_sum
            st.session_state['w_char5'] = w_char5 / weights_sum
            st.session_state['w_word'] = w_word / weights_sum
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Display Filters")
min_similarity_filter = st.sidebar.slider("Min similarity to display (%)", 0, 100, 0, 5)

uploaded_file = st.file_uploader("Upload Activity JSON", type=["json"])

if uploaded_file:
    try:
        data = json.load(uploaded_file)
    except json.JSONDecodeError:
        st.error("Invalid JSON file.")
        st.stop()
    
    if not isinstance(data, list):
        st.error("JSON must be a list of activity responses.")
        st.stop()
    
    st.success(f"Loaded {len(data)} records from JSON")
    
    if st.button("Run Plagiarism Analysis", type="primary"):
        # Check if all weights are zero
        if weights_sum == 0:
            st.error("⚠️ Cannot run analysis: All weights are 0. Please set at least one weight greater than 0.")
            st.stop()
        
        # Auto-normalize weights if needed
        normalized_weights = weights.copy()
        normalization_message = None
        
        if abs(weights_sum - 1.0) > 0.001:
            # Normalize proportionally
            normalized_weights = {
                'tfidf': w_tfidf / weights_sum,
                'char3': w_char3 / weights_sum,
                'char5': w_char5 / weights_sum,
                'word_jaccard': w_word / weights_sum
            }
            
            # Create normalization message
            changes = []
            if w_tfidf > 0:
                changes.append(f"TF-IDF: {w_tfidf:.2f}→{normalized_weights['tfidf']:.2f}")
            if w_char3 > 0:
                changes.append(f"Char-3: {w_char3:.2f}→{normalized_weights['char3']:.2f}")
            if w_char5 > 0:
                changes.append(f"Char-5: {w_char5:.2f}→{normalized_weights['char5']:.2f}")
            if w_word > 0:
                changes.append(f"Word: {w_word:.2f}→{normalized_weights['word_jaccard']:.2f}")
            
            normalization_message = f"🔧 **Weights auto-normalized:** {' | '.join(changes)}"
            st.info(normalization_message)
        
        with st.spinner("Analyzing... This may take a while for large datasets."):
            detailed_rows, skipped = run_plagiarism_analysis(data, threshold, normalized_weights)
        
        if not detailed_rows:
            st.warning("No valid pairs found for comparison.")
            st.stop()
        
        # Store in session state to persist after reruns
        st.session_state['results_df'] = pd.DataFrame(detailed_rows)
        st.session_state['skipped'] = skipped
        st.session_state['threshold'] = threshold
        st.session_state['weights'] = normalized_weights
        st.session_state['normalization_msg'] = normalization_message

# Display results if available
if 'results_df' in st.session_state:
    df = st.session_state['results_df']
    skipped = st.session_state['skipped']
    analysis_threshold = st.session_state.get('threshold', threshold)
    
    # Sort by ensemble score descending
    df_sorted = df.sort_values("Ensemble (%)", ascending=False)
    flagged_df = df_sorted[df_sorted["Ensemble (%)"] >= threshold]
    
    # ==================== SUMMARY SECTION ====================
    st.header("Summary")
    
    # Display normalization message if weights were auto-normalized
    normalization_msg = st.session_state.get('normalization_msg')
    if normalization_msg:
        st.info(normalization_msg)
    
    # Display weights used for this analysis
    analysis_weights = st.session_state.get('weights', weights)
    with st.expander("📊 Ensemble Weights Used", expanded=False):
        wcol1, wcol2, wcol3, wcol4 = st.columns(4)
        wcol1.metric("TF-IDF", f"{analysis_weights['tfidf']:.3f}")
        wcol2.metric("Char-3", f"{analysis_weights['char3']:.3f}")
        wcol3.metric("Char-5", f"{analysis_weights['char5']:.3f}")
        wcol4.metric("Word Jaccard", f"{analysis_weights['word_jaccard']:.3f}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Records", len(data) if 'data' in dir() else "N/A")
    col2.metric("Total Pairs", len(df))
    col3.metric("Flagged Pairs", len(flagged_df), delta=f"{len(flagged_df)/len(df)*100:.1f}%" if len(df) > 0 else "0%")
    col4.metric("Max Similarity", f"{df['Ensemble (%)'].max():.1f}%")
    col5.metric("Skipped", len(skipped))
    
    # Activity-wise summary
    st.subheader("Activity-wise Summary")
    activity_summary = df.groupby(['Instance ID', 'Activity Name']).agg({
        'Ensemble (%)': ['count', 'max', 'mean']
    }).reset_index()
    activity_summary.columns = ['Instance ID', 'Activity Name', 'Total Pairs', 'Max Similarity (%)', 'Avg Similarity (%)']
    
    # Add flagged pairs count based on threshold
    def count_flagged(group):
        return (group['Ensemble (%)'] >= threshold).sum()
    
    flagged_counts = df.groupby(['Instance ID', 'Activity Name']).apply(count_flagged).reset_index()
    flagged_counts.columns = ['Instance ID', 'Activity Name', 'Flagged Pairs']
    
    activity_summary = activity_summary.merge(flagged_counts, on=['Instance ID', 'Activity Name'])
    activity_summary['Avg Similarity (%)'] = activity_summary['Avg Similarity (%)'].round(2)
    activity_summary = activity_summary.sort_values('Max Similarity (%)', ascending=False)
    
    st.dataframe(activity_summary, use_container_width=True, height=200)
    
    # ==================== FULL RESULTS TABLE ====================
    st.header("All Similarity Results")
    st.caption(f"Showing all pairs sorted by similarity (highest first). Threshold: {threshold}%")
    
    # Apply display filters
    display_df = df_sorted.copy()
    if min_similarity_filter > 0:
        display_df = display_df[display_df["Ensemble (%)"] >= min_similarity_filter]
    
    display_cols = ["Instance ID", "Activity Name", "User 1", "User 2", 
                   "TF-IDF (%)", "Char-3 (%)", "Char-5 (%)", "Word Jaccard (%)",
                   "Exact Copy", "Ensemble (%)"]
    
    # Color coding for the dataframe
    def highlight_similarity(row):
        if row['Ensemble (%)'] >= threshold:
            return ['background-color: #ffcccc'] * len(row)
        elif row['Ensemble (%)'] >= 50:
            return ['background-color: #fff3cd'] * len(row)
        return [''] * len(row)
    
    st.dataframe(
        display_df[display_cols].style.apply(highlight_similarity, axis=1),
        use_container_width=True,
        height=400
    )
    
    st.info(f"Displaying {len(display_df)} of {len(df)} pairs")
    
    # ==================== DETAILED COMPARISONS ====================
    st.header("Detailed Comparisons with Highlighted Text")
    st.caption("Click on any pair to see the full text comparison with matching parts highlighted in yellow")
    
    # Pagination for large datasets
    pairs_per_page = 20
    total_display = len(display_df)
    total_pages = max(1, (total_display + pairs_per_page - 1) // pairs_per_page)
    
    if total_pages > 1:
        page = st.selectbox(f"Page (showing {pairs_per_page} pairs per page)", range(1, total_pages + 1), index=0)
    else:
        page = 1
    
    start_idx = (page - 1) * pairs_per_page
    end_idx = min(start_idx + pairs_per_page, total_display)
    
    st.markdown(f"**Showing pairs {start_idx + 1} to {end_idx} of {total_display}**")
    
    # Display pairs with highlighting
    for i, (idx, row) in enumerate(display_df.iloc[start_idx:end_idx].iterrows()):
        similarity = row['Ensemble (%)']
        
        # Color-coded label based on similarity
        if similarity >= threshold:
            status_emoji = "🔴"
        elif similarity >= 50:
            status_emoji = "🟡"
        else:
            status_emoji = "🟢"
        
        user1_short = row['User 1'][:12] + "..." if len(row['User 1']) > 12 else row['User 1']
        user2_short = row['User 2'][:12] + "..." if len(row['User 2']) > 12 else row['User 2']
        
        label = f"{status_emoji} **{similarity}%** | {user1_short} vs {user2_short} | {row['Activity Name'][:30]}"
        
        with st.expander(label, expanded=(i == 0 and page == 1)):  # Expand first item on first page
            # Scores row
            score_cols = st.columns(5)
            score_cols[0].metric("TF-IDF", f"{row['TF-IDF (%)']}%")
            score_cols[1].metric("Char-3", f"{row['Char-3 (%)']}%")
            score_cols[2].metric("Char-5", f"{row['Char-5 (%)']}%")
            score_cols[3].metric("Word Jaccard", f"{row['Word Jaccard (%)']}%")
            score_cols[4].metric("Ensemble", f"{row['Ensemble (%)']}%")
            
            st.markdown(f"**Activity:** {row['Activity Name']} | **Instance:** {row['Instance ID']} | **Activity ID:** {row['Activity ID']}")
            
            if row['Exact Copy'] == "Yes":
                st.error("⚠️ EXACT COPY DETECTED")
            
            # Highlight matching parts
            h1, h2 = highlight_common_sequences(row["Answer 1"], row["Answer 2"], min_highlight_len)
            
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**User 1:** `{row['User 1']}`")
                st.markdown(
                    f'<div style="background-color: #f0f0f0; padding: 15px; border-radius: 5px; border-left: 4px solid #1f77b4;">{h1}</div>',
                    unsafe_allow_html=True
                )
            with c2:
                st.markdown(f"**User 2:** `{row['User 2']}`")
                st.markdown(
                    f'<div style="background-color: #f0f0f0; padding: 15px; border-radius: 5px; border-left: 4px solid #ff7f0e;">{h2}</div>',
                    unsafe_allow_html=True
                )
    
    # ==================== DOWNLOAD SECTION ====================
    st.header("Download Results")
    
    col1, col2 = st.columns(2)
    with col1:
        csv_all = df_sorted.to_csv(index=False)
        st.download_button(
            "📥 Download All Results (CSV)",
            data=csv_all,
            file_name="plagiarism_all_results.csv",
            mime="text/csv"
        )
    
    with col2:
        if len(flagged_df) > 0:
            csv_flagged = flagged_df.to_csv(index=False)
            st.download_button(
                "🚨 Download Flagged Only (CSV)",
                data=csv_flagged,
                file_name="plagiarism_flagged_results.csv",
                mime="text/csv"
            )
    
    # Show skipped entries
    if skipped:
        with st.expander(f"📋 View {len(skipped)} Skipped Records"):
            st.dataframe(pd.DataFrame(skipped), use_container_width=True)
