"""
app.py — AcityPal Streamlit Application — WOW Edition
=======================================================
Full RAG system with:
  • Animated Ghana-flag landing page
  • Glassy chat interface
  • Structured response cards + vote bar charts
  • Smart sidebar (history groups, pin, export)
  • Query intent router (ELECTION / BUDGET / COMPARE)
  • Bottom-centered chat input with gradient send button
  • Retrieval settings + region filter in sidebar
  • Parts A–G exam coverage hidden in backend
"""

from pathlib import Path
from datetime import datetime, timedelta
import base64
import json
import re
import time
import streamlit as st
import streamlit.components.v1 as components
import html
from src.pipeline import RAGPipeline

st.set_page_config(
    page_title="AcityPal — Ghana RAG System",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── GLOBAL AUTO-SCROLL JAVASCRIPT (injected via HTML) ───────────────────────────
st.markdown("""
<script id="auto-scroll-script">
window.addEventListener('DOMContentLoaded', function() {
    console.log('[Auto-scroll] DOMContentLoaded fired');
    
    function scrollToBottom() {
        console.log('[Auto-scroll] scrollToBottom called');
        var container = document.querySelector('.chat-scrollable-container');
        console.log('[Auto-scroll] Container found:', !!container);
        
        if (container) {
            console.log('[Auto-scroll] Container scrollHeight:', container.scrollHeight);
            console.log('[Auto-scroll] Container clientHeight:', container.clientHeight);
            container.scrollTop = container.scrollHeight;
            console.log('[Auto-scroll] Scrolled to bottom');
        }
        
        window.scrollTo(0, document.body.scrollHeight);
    }
    
    // Initial scroll
    setTimeout(scrollToBottom, 100);
    
    // Watch for DOM changes
    var observer = new MutationObserver(function() {
        console.log('[Auto-scroll] DOM changed');
        scrollToBottom();
    });
    observer.observe(document.body, { childList: true, subtree: true });
    
    // Periodic scroll
    setInterval(scrollToBottom, 500);
});
</script>
""", unsafe_allow_html=True)

# ── UNIVERSAL PRE-PAINT KILL-SWITCH (Prevents all flashes) ─────────────────────
st.markdown("""
<style>
    /* 1. Global Silence - hide app and stop scrolling until fully ready */
    header, [data-testid="stHeader"], footer { display: none !important; }
    .stApp {{ 
        background: linear-gradient(-45deg, #006B3F, #004d2d, #CE1126, #8b0000, #FCD116, #b8960a, #006B3F, #CE1126) !important;
        background-size: 500% 500% !important;
        animation: ghanaGrad 12s ease infinite !important;
        visibility: hidden; 
    }}
    .block-container { visibility: hidden; }
    
    /* 2. Global Keyframes */
    @keyframes ghanaGrad {
      0%   { background-position: 0% 50%; }
      50%  { background-position: 100% 50%; }
      100% { background-position: 0% 50%; }
    }
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

def find_logo_path():
    root = Path(__file__).resolve().parent
    explicit = root / "images" / "acitylogo.png"
    if explicit.exists():
        return str(explicit)
    candidates = []
    for folder in ["image", "images", "assets", "static"]:
        p = root / folder
        if p.exists():
            for ext in ("png", "jpg", "jpeg", "webp"):
                candidates.extend(sorted(p.glob(f"*.{ext}")))
    return str(candidates[0]) if candidates else None


def build_logo_data_uri(path):
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    suffix = p.suffix.lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(suffix, "image/png")
    encoded = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


@st.cache_resource(show_spinner="Loading AcityPal knowledge base…")
def get_pipeline():
    return RAGPipeline()

def init_state():
    defaults = {
        "started": False,
        "show_sidebar": False,
        "top_k": 5,
        "use_hybrid": True,
        "conversations": [],
        "active_chat_idx": None,
        "theme_mode": "light",
        "active_tab": "chat",
        "eval_results": None,
        "chunking_data": None,
        "failure_data": None,
        "prompt_cmp_data": None,
        "rag_vs_llm_data": None,
        "region_filter": "All Regions",
        "prompt_variant_sel": "hybrid",
        "is_typing": False,
        "has_chart": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def create_new_chat():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    chat = {"title": "New Chat", "created_at": now, "messages": [], "pinned": False}
    st.session_state.conversations.insert(0, chat)
    st.session_state.active_chat_idx = 0


def get_active_chat():
    idx = st.session_state.active_chat_idx
    if idx is None or idx >= len(st.session_state.conversations):
        return None
    return st.session_state.conversations[idx]


# ── Query Intent Router ────────────────────────────────────────────────────────

GHANA_REGIONS = [
    "All Regions", "Greater Accra", "Ashanti", "Western", "Central", "Eastern",
    "Volta", "Northern", "Upper East", "Upper West", "Brong-Ahafo",
    "Ahafo", "Bono East", "Oti", "Savannah", "North East", "Western North",
]

def classify_query_intent(query: str) -> str:
    """Classify into ELECTION, BUDGET, or COMPARE for routing."""
    q = query.lower()
    election_terms = {"votes", "voted", "won", "winner", "candidate", "npp", "ndc",
                      "election", "presidential", "region", "constituency", "results",
                      "polling", "ballots", "akufo", "mahama", "bawumia", "awuni"}
    budget_terms = {"budget", "fiscal", "deficit", "revenue", "expenditure", "tax",
                    "gdp", "policy", "borrowing", "infrastructure", "allocation",
                    "allocated", "spending", "investment", "growth", "debt", "surplus", 
                    "ghc", "gh¢", "cost", "amount", "million", "billion"}
    compare_terms = {"compare", "versus", "vs", "difference", "between", "both",
                     "relative", "than", "higher", "lower", "more", "less"}
    words = set(re.findall(r"\w+", q))
    e = len(words & election_terms)
    b = len(words & budget_terms)
    c = len(words & compare_terms)
    if c > 0 and (e > 0 or b > 0):
        return "COMPARE"
    if e > b:
        return "ELECTION"
    return "BUDGET"


def parse_vote_data(retrieved_docs: list) -> list:
    """Extract candidate vote data from CSV retrieved chunks for charting."""
    vote_pattern = re.compile(
        r"-\s*([^(]+)\(([^)]+)\):\s*([\d,]+)\s*votes", re.IGNORECASE
    )
    candidates = {}
    for d in retrieved_docs:
        if d.get("metadata", {}).get("source") != "csv":
            continue
        for name, party, votes_str in vote_pattern.findall(d.get("text", "")):
            name = name.strip()
            party = party.strip()
            try:
                v = int(votes_str.replace(",", ""))
            except ValueError:
                continue
            key = f"{name} ({party})"
            candidates[key] = candidates.get(key, 0) + v
    if not candidates:
        return []
    return sorted(
        [{"label": k, "votes": v} for k, v in candidates.items()],
        key=lambda x: x["votes"], reverse=True
    )[:8]


# ── Global CSS ─────────────────────────────────────────────────────────────────



def inject_global_styles(allow_scroll=True):
    is_dark = st.session_state.theme_mode == "dark"
    bg      = "#0b1020" if is_dark else "#f0f4f8"
    surface = "#0f172a" if is_dark else "#ffffff"
    text_1  = "#e5e7eb" if is_dark else "#111827"
    text_2  = "#94a3b8" if is_dark else "#6b7280"
    line    = "#1e293b" if is_dark else "#e2e8f0"
    bubble_u = "rgba(111,66,255,0.85)" if is_dark else "rgba(111,66,255,0.9)"
    bubble_b = "rgba(15,45,92,0.85)" if is_dark else "rgba(239,246,255,0.85)"

    sidebar_active = st.session_state.get("show_sidebar", False)
    input_left = "62.5%" if sidebar_active else "50%"
    input_max_width = "800px" if sidebar_active else "900px"
    ov_y = "auto" if allow_scroll else "hidden"

    st.markdown(f"""
<style>

/* Chat input shift logic */
div[data-testid="stChatInput"] {{
  left: {input_left} !important;
  max-width: {input_max_width} !important;
  transition: left 0.3s ease !important;
}}


/* ── X Button with hover tooltip (Close Sidebar) ── */
.sidebar-close-btn {{
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  font-size: 20px;
  font-weight: 600;
  color: var(--text-2);
  background: transparent;
  border: none;
  width: 32px;
  height: 32px;
  border-radius: 8px;
  transition: all 0.2s ease;
}}

.sidebar-close-btn:hover {{
  background: rgba(206,17,38,0.1);
  color: #CE1126;
}}

/* Tooltip text */
.sidebar-close-btn .tooltip-text {{
  visibility: hidden;
  background-color: #CE1126;
  color: white;
  text-align: center;
  padding: 5px 10px;
  border-radius: 6px;
  position: absolute;
  z-index: 1;
  bottom: 125%;
  left: 50%;
  transform: translateX(-50%);
  white-space: nowrap;
  font-size: 12px;
  font-weight: 500;
  opacity: 0;
  transition: opacity 0.3s;
  pointer-events: none;
}}

.sidebar-close-btn .tooltip-text::after {{
  content: "";
  position: absolute;
  top: 100%;
  left: 50%;
  margin-left: -5px;
  border-width: 5px;
  border-style: solid;
  border-color: #CE1126 transparent transparent transparent;
}}

.sidebar-close-btn:hover .tooltip-text {{
  visibility: visible;
  opacity: 1;
}}

/* ── Dark Mode Toggle (emoji only, red color) ── */
.darkmode-toggle-emoji {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  font-size: 20px;
  background: transparent;
  border: none;
  width: 32px;
  height: 32px;
  border-radius: 8px;
  transition: all 0.2s ease;
  color: #CE1126 !important;
}}

.darkmode-toggle-emoji:hover {{
  background: rgba(206,17,38,0.15);
  transform: scale(1.05);
}}

/* Header right side buttons container */
.sidebar-header-actions {{
  display: flex;
  align-items: center;
  gap: 4px;
}}

/* ── TWO-SCREEN LAYOUT STRUCTURE ── */
/* Scoped to main container to avoid breaking top-level Streamlit elements */
[data-testid="stMainBlockContainer"] {{
  padding: 0 !important;
}}

/* Sidebar panel - shown when sidebar is open */
.sidebar-panel {{
  height: 100vh !important;
  overflow-y: auto !important;
  background: var(--surface) !important;
  padding: 12px !important;
  border-right: 5px solid #CE1126 !important;
  box-shadow: 8px 0 25px rgba(0, 0, 0, 0.25) !important;
}}

/* ── FINAL 3D SIDEBAR FORCE - EDGE TO EDGE ── */
[data-testid="stAppViewBlockContainer"] {{
  padding-top: 0 !important;
  padding-bottom: 0 !important;
  max-width: 100% !important;
}}

/* Ensure the sidebar column is flush against the edges */
[data-testid="column"]:has(.sidebar-panel-fix) {{
  position: relative !important;
  overflow: visible !important;
  background: var(--surface) !important;
  height: 100vh !important;
  margin-top: 0 !important;
}}

/* The physical gradient line and shadow - FIXED to touch edges */
.sidebar-panel-fix-edge {{
  position: fixed !important;
  top: 12px !important; /* Starts below the top gradient bar */
  bottom: 0 !important;
  left: calc(100% * 0.325) !important; /* 1.3/4.0 ratio */
  width: 4px !important;
  background: linear-gradient(180deg, #CE1126 0%, #FCD116 50%, #006B3F 100%) !important;
  box-shadow: 15px 0 45px rgba(0,0,0,0.3) !important;
  z-index: 10000 !important;
}}

/* Fallback for the edge div if relative works better */
[data-testid="column"]:has(.sidebar-panel-fix) .sidebar-panel-fix-edge {{
  position: absolute !important;
  right: 0 !important;
  left: auto !important;
}}

/* ── GUARANTEED SIDEBAR CENTERING & CLEARANCE ── */

/* ── GUARANTEED SIDEBAR CENTERING & CLEARANCE ── */
/* Target the main containers with a MASSIVE 80px gutter */
[data-testid="column"]:first-child [data-testid="stVerticalBlock"],
[data-testid="column"]:first-child [data-testid="stHorizontalBlock"],
[data-testid="column"]:first-child .stVerticalBlock,
[data-testid="column"]:first-child .element-container {{
  padding-left: 80px !important;
  padding-right: 80px !important;
  padding-top: 12px !important; /* Touch the bottom of the 12px gradient bar */
  margin-top: 0 !important;
  margin-right: 0 !important;
  box-sizing: border-box !important;
}}

/* Target specific widgets that often ignore parent padding */
[data-testid="column"]:first-child .stSlider, 
[data-testid="column"]:first-child .stSelectbox,
[data-testid="column"]:first-child [data-testid="stHelpIcon"] {{
  margin-right: 80px !important;
  margin-left: 80px !important;
}}

.sidebar-internal-content {{
  padding-right: 0 !important;
  margin-top: 0 !important;
  padding-top: 0 !important;
}}

/* Sidebar container with border */
.sidebar-border-container {{
  border-right: 10px solid rgba(206, 17, 38, 0.8) !important;
  height: 100% !important;
  padding-right: 2px !important;
}}

/* Style the Streamlit container inside the sidebar column */
[data-testid="column"]:first-child:has(.sidebar-panel-fix) > div > div {{
  height: 100vh !important;
  overflow-y: auto !important;
  overflow-x: hidden !important;
  padding-bottom: 50px !important; /* Extra space at bottom */
}}

/* Custom Scrollbar for Sidebar only */
[data-testid="column"]:first-child:has(.sidebar-panel-fix) > div > div::-webkit-scrollbar {{
  width: 5px !important;
}}

[data-testid="column"]:first-child:has(.sidebar-panel-fix) > div > div::-webkit-scrollbar-thumb {{
  background: linear-gradient(180deg, #CE1126, #FCD116, #006B3F) !important;
  border-radius: 10px !important;
}}

[data-testid="column"]:first-child:has(.sidebar-panel-fix) > div > div::-webkit-scrollbar-track {{
  background: rgba(0,0,0,0.05) !important;
}}

/* Chat column styling */
[data-testid="column"]:last-child:has(.chat-panel-fix) {{
  height: 100vh !important;
  overflow-y: hidden !important;
  padding: 0 !important;
}}

[data-testid="column"]:last-child:has(.chat-panel-fix).has-scroll {{
  overflow-y: auto !important;
}}

[data-testid="column"]:last-child:has(.chat-panel-fix) > div > div {{
  height: 100vh !important;
  overflow-y: hidden !important;
}}

[data-testid="column"]:last-child:has(.chat-panel-fix).has-scroll > div > div {{
  overflow-y: auto !important;
}}

/* Hide the structural marker div for chat only (sidebar marker is used for visual border) */
.chat-panel-fix {{
  display: none !important;
}}

/* Main chat area - centered when sidebar closed, left when open */
[data-testid="column"]:last-child {{
  flex: 1 !important;
  height: 100vh !important;
  overflow-y: hidden !important;
  position: relative !important;
  padding: 20px 40px !important;
}}

[data-testid="column"]:last-child.has-scroll {{
  overflow-y: auto !important;
}}

/* Chat panel - right side */
.chat-panel {{
  height: 100vh !important;
  overflow-y: hidden !important;
  padding: 0 20px !important;
}}

.chat-panel.has-scroll {{
  overflow-y: auto !important;
}}

/* When sidebar is open, limit width and center the chat content */
.chat-panel {{
  max-width: 900px !important;
  margin: 0 auto !important;
}}

/* Reduce top padding to bring header to top edge */
.main .block-container {{
  padding-top: 0px !important;
  padding-bottom: 0 !important;
}}

/* ── CENTERED CHAT INPUT (FIXED) ── */
div[data-testid="stChatInput"] {{
  position: fixed !important;
  bottom: 20px !important;
  /* left and max-width are handled dynamically in Python */
  transform: translateX(-50%) !important;
  width: auto !important;
  min-width: 600px !important;
  z-index: 9999;
  display: flex !important;
  align-items: center !important;
  gap: 12px !important;
  background: transparent !important;
}}




/* ── GLASSMORPHISM INPUT BOX ── */
div[data-testid="stChatInput"] textarea {{
  border-radius: 28px !important;
  padding: 16px 24px !important;
  border: 2px solid rgba(206, 17, 38, 0.12) !important;
  box-shadow:
    0 8px 32px rgba(0, 0, 0, 0.08),
    0 2px 12px rgba(0, 0, 0, 0.04),
    inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
  font-size: 0.96rem !important;
  font-weight: 500 !important;
  background: rgba(255, 255, 255, 0.92) !important;
  backdrop-filter: blur(24px) saturate(200%) !important;
  -webkit-backdrop-filter: blur(24px) saturate(200%) !important;
  flex: 1 !important;
  margin: 0 !important;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}}

/* Focus state with red glow */
div[data-testid="stChatInput"] textarea:focus {{
  border-color: rgba(206, 17, 38, 0.4) !important;
  box-shadow:
    0 12px 40px rgba(206, 17, 38, 0.18),
    0 4px 16px rgba(206, 17, 38, 0.12),
    0 0 0 4px rgba(206, 17, 38, 0.08),
    inset 0 1px 0 rgba(255, 255, 255, 0.9) !important;
  background: rgba(255, 255, 255, 0.98) !important;
  transform: translateY(-1px) !important;
}}



/* ── SEND BUTTON WITH VISIBLE EDGE, BORDER AND SHADOW ── */
div[data-testid="stChatInput"] button {{
  background: linear-gradient(135deg, #CE1126 0%, #8B0000 100%) !important;
  color: white !important;
  border: 2px solid rgba(255, 255, 255, 0.25) !important;
  border-radius: 28px !important;
  padding: 16px 32px !important;
  font-size: 0.95rem !important;
  font-weight: 700 !important;
  box-shadow:
    0 4px 16px rgba(206, 17, 38, 0.35),
    0 2px 8px rgba(206, 17, 38, 0.2),
    inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
  margin: 0 !important;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
  min-width: 100px !important;
  cursor: pointer !important;
  letter-spacing: 0.02em;
}}

div[data-testid="stChatInput"] button:hover {{
  transform: translateY(-2px) scale(1.02) !important;
  box-shadow:
    0 8px 24px rgba(206, 17, 38, 0.45),
    0 4px 12px rgba(206, 17, 38, 0.3),
    inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
  border: 2px solid rgba(255, 255, 255, 0.35) !important;
}}

/* Active/pressed state */
div[data-testid="stChatInput"] button:active {{
  transform: translateY(0) scale(0.98) !important;
  box-shadow:
    0 2px 8px rgba(206, 17, 38, 0.3),
    0 1px 4px rgba(206, 17, 38, 0.2) !important;
}}

/* ── SETTINGS SELECTBOX DROPDOWNS (Ensure they work properly inside container) ── */
/* Make selectbox dropdowns visible and clickable */
.settings-card div[data-testid="stSelectbox"] {{
  position: relative;
  z-index: 10;
}}

/* Ensure dropdown menu appears properly */
.settings-card div[data-testid="stSelectbox"] div[data-baseweb="popover"] {{
  z-index: 9999 !important;
  position: absolute !important;
}}

/* Force dropdown to expand and be visible */
.settings-card div[data-testid="stSelectbox"] ul {{
  max-height: 300px !important;
  overflow-y: auto !important;
}}

/* ── HORIZONTAL ALIGNMENT FOR INPUT + BUTTON ── */
.stChatInputContainer {{
  display: flex !important;
  flex-direction: row !important;
  align-items: center !important;
  gap: 12px !important;
  width: 100% !important;
}}

/* Force the chat input to be a flex row container */
[data-testid="stChatInput"] > div {{
  display: flex !important;
  flex-direction: row !important;
  align-items: center !important;
  gap: 12px !important;
  width: 100% !important;
}}

/* FIX chat input position */
div[data-testid="stChatInput"] {{
  position: fixed !important;
  bottom: 6px !important;   /* 🔥 reduce this to move closer to edge */
  z-index: 9999;
  padding: 0 16px;
}}



.input-wrapper {{
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 2px 0 0;
  z-index: 500;
  transition: all 0.3s ease;
}}




/* REMOVE Streamlit’s default top spacing completely */
[data-testid="stAppViewContainer"] {{ 
  padding-top: 0 !important;
}}

[data-testid="stAppViewContainer"] > .main {{
  padding-top: 0 !important;
  margin-top: 0 !important;
}}

/* Extra safety override (this is often the real culprit) */
[data-testid="stMainBlockContainer"] {{
  padding-top: 0 !important;
  margin-top: 0 !important;
}}



/* animation */
@keyframes fadeInOut {{
  0% {{ opacity: 0; transform: translateX(-50%) translateY(-5px); }}
  20% {{ opacity: 1; transform: translateX(-50%) translateY(0); }}
  80% {{ opacity: 1; }}
  100% {{ opacity: 0; }}
}}

div[data-testid="stHorizontalBlock"] {{
  padding-left: 8px !important;
  padding-right: 8px !important;
  gap: 6px !important;
}}

.header-flex-marker {{
  margin-top: 0 !important;
  padding-top: 0 !important;
  height: 0 !important;
}}

[data-testid="stHeader"] {{ 
  height: 0px !important;
  visibility: hidden !important;
}}

.block-container {{
 padding-left: 8px !important;
  padding-right: 8px !important;
}}

.header-flex-marker,
[data-testid="column"] {{
  display: flex !important;
  align-items: center !important;
  justify-content: flex-start !important;
}}  

[data-testid="column"] {{
  flex: 0 !important;
  width: auto !important;
}}

div[data-testid="stHorizontalBlock"] {{
  gap: 8px !important;
}}

.chat-header-bar {{
  top: 0;
  z-index: 9999;
  background: var(--surface);
}}

.chat-header-bar,
.header-flex-marker {{
  margin-top: 0 !important;
  padding-top: 0 !important;
}}



/* Aggressively push header to top */
[data-testid="stVerticalBlock"]:has(.header-push-marker) {{
  margin-top: -55px !important;
}}

.header-push-marker {{
  display: none !important;
}}

.top-header .left {{
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 800;
}}

.top-header .brand {{
  color: #CE1126;
  font-weight: 900;
}}  

.top-header .right {{
  display: flex;
  gap: 10px;
  align-items: center;
}}

.icon-btn {{
  font-size: 18px;
  background: transparent;
  border: none;
  cursor: pointer;
  color: var(--text-1);
}}


/* Enable page scroll */
html, body {{
  overflow-y: {ov_y} !important;
  overflow-x: hidden !important;
  height: auto !important;
  min-height: 100vh !important;
}}

/* Allow chat to scroll */
.block-container {{
  overflow-y: {ov_y} !important;
  max-height: 100vh !important;
}}

/* Enable Streamlit containers to scroll */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {{
  overflow-y: {ov_y} !important;
  overflow-x: hidden !important;
  height: auto !important;
  min-height: 100vh !important;
}}



@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {{
  --bg:         {bg};
  --surface:    {surface};
  --text-1:     {text_1};
  --text-2:     {text_2};
  --line:       {line};
  --accent-1:   #6f42ff;
  --accent-2:   #00a2ff;
  --red-gh:     #CE1126;
  --gold-gh:    #FCD116;
  --green-gh:   #006B3F;
  --bubble-u:   {bubble_u};
  --bubble-b:   {bubble_b};
}}

/* ── Reset ── */
body {{
  overflow-x: hidden !important;
}}


#MainMenu, footer, header, [data-testid="stHeader"] {{ display:none !important; }}
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {{
  background: var(--bg) !important;
  animation: none !important; /* Stop Ghana animation in chat mode */
  color: var(--text-1) !important;
  font-family: 'Inter', system-ui, sans-serif !important;
  margin: 0 !important;
  overflow-x: hidden !important;
}}

  overflow-x: hidden !important;
}}

/* ── RAG DETAILS PREMIUM BUBBLE EXPANDER ── */
.rag-bubble-wrapper {{
    display: flex;
    justify-content: flex-end;
    margin-top: -10px;
    margin-bottom: 20px;
    padding-right: 1rem;
}}

/* Global expander override - more subtle by default */
[data-testid="stExpander"] {{
    border: 1px solid var(--line) !important;
    background: transparent !important;
    border-radius: 8px !important;
    margin-bottom: 1rem !important;
}}

/* ── RAG DETAILS BUBBLE (Premium Floating Card) ── */
.rag-bubble-wrapper {{
    display: flex !important;
    justify-content: center !important;
    width: 100% !important;
    margin: 20px 0 28px !important;
    clear: both;
}}

.rag-bubble-wrapper [data-testid="stExpander"] {{
    width: auto !important;
    min-width: 360px !important;
    max-width: 860px !important;
    border: none !important;
    border-radius: 18px !important;
    background: var(--surface) !important;
    box-shadow:
        0 0 0 1.5px rgba(206,17,38,0.22),
        0 8px 32px rgba(206,17,38,0.10),
        0 24px 64px rgba(0,0,0,0.10) !important;
    overflow: hidden !important;
    margin: 0 auto !important;
    position: relative !important;
    transition: box-shadow 0.3s ease, transform 0.3s ease !important;
}}

/* Ghana flag gradient top stripe */
.rag-bubble-wrapper [data-testid="stExpander"]::before {{
    content: '' !important;
    display: block !important;
    position: absolute !important;
    top: 0 !important; left: 0 !important; right: 0 !important;
    height: 3px !important;
    background: linear-gradient(90deg, #CE1126 0%, #FCD116 50%, #006B3F 100%) !important;
    z-index: 2 !important;
    border-radius: 18px 18px 0 0 !important;
}}

.rag-bubble-wrapper [data-testid="stExpander"]:hover {{
    box-shadow:
        0 0 0 2px rgba(206,17,38,0.4),
        0 12px 40px rgba(206,17,38,0.15),
        0 32px 80px rgba(0,0,0,0.13) !important;
    transform: translateY(-2px) !important;
}}

/* The trigger button (Expander Summary) */
[data-testid="stExpanderSummary"] {{
    background: linear-gradient(
        135deg,
        rgba(206,17,38,0.06) 0%,
        rgba(252,209,22,0.04) 50%,
        rgba(0,107,63,0.04) 100%
    ) !important;
    color: var(--text-1) !important;
    padding: 14px 20px !important;
    border-radius: 0 !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.01em !important;
    border: none !important;
    border-bottom: 1px solid rgba(206,17,38,0.10) !important;
    box-shadow: none !important;
    transition: background 0.25s ease !important;
    animation: none !important;
    width: 100% !important;
    min-width: unset;
}}

[data-testid="stExpanderSummary"]:hover {{
    background: linear-gradient(
        135deg,
        rgba(206,17,38,0.10) 0%,
        rgba(252,209,22,0.07) 50%,
        rgba(0,107,63,0.06) 100%
    ) !important;
    transform: none !important;
    box-shadow: none !important;
}}

/* Style the "RAG Details" text label */
[data-testid="stExpanderSummary"] p,
[data-testid="stExpanderSummary"] span {{
    font-weight: 700 !important;
    color: var(--text-1) !important;
}}

/* The expand arrow icon */
[data-testid="stExpanderSummary"] svg {{
    fill: #CE1126 !important;
    width: 18px !important;
    height: 18px !important;
}}

/* The details panel */
[data-testid="stExpanderDetails"] {{
    background: transparent !important;
    padding: 24px !important;
    width: 100% !important;
    max-width: 100% !important;
    border: none !important;
    margin: 0 !important;
}}

/* Feedback thumbs — flush against bottom of RAG card */
.feedback-row-wrapper {{
    display: flex !important;
    justify-content: center !important;
    width: 100% !important;
    position: absolute !important;
    bottom: -10px !important;
    left: 0 !important;
    z-index: 100 !important;
    margin: 0 !important;
}}

.rag-bubble-wrapper {{
    position: relative !important;
}}

.feedback-row-wrapper > div[data-testid="stHorizontalBlock"] {{
    max-width: 860px !important;
    width: 100% !important;
    padding-left: 4px !important;
    margin: 0 !important;
}}

/* Special override for RAG wrapper inside expander details to avoid double-borders */
.rag-bubble-wrapper [data-testid="stExpanderDetails"] {{
    border-top: 1px solid rgba(255, 59, 59, 0.2) !important;
}}

/* ── CHUNK DETAILS "WOW" STYLING ── */
.rag-bubble-wrapper .chunk-details-inner {{
    background: rgba(15, 15, 15, 0.4) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    margin-top: 10px !important;
}}

.rag-bubble-container-inner {{
    display: flex !important;
    gap: 12px !important;
    flex-wrap: wrap !important;
    margin-bottom: 15px !important;
}}

.chunk-metric-card {{
    background: #FFFFFF !important; /* White container */
    border: 2px solid #ff3b3b !important; /* Red borderline */
    border-radius: 12px !important;
    padding: 12px 14px !important;
    min-width: 120px !important;
    flex: 1 !important;
    text-align: center !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
    transition: transform 0.3s ease !important;
}}

.chunk-metric-card:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(255, 59, 59, 0.2) !important;
}}

.chunk-metric-label {{
    font-size: 0.85rem !important;
    color: #000000 !important; /* Bold Black heading */
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    margin-bottom: 6px !important;
    font-weight: 900 !important;
}}

.chunk-metric-value {{
    font-size: 1.2rem !important;
    color: #ff3b3b !important; /* Red numbers */
    font-weight: 900 !important;
    text-shadow: none !important;
}}

.rag-bubble-wrapper .chunk-text-container {{
    background: rgba(0, 0, 0, 0.2) !important;
    border-left: 4px solid #ff3b3b !important;
    border-top: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 8px !important;
    padding: 18px !important;
    font-weight: 800 !important;
    color: #f0f0f0 !important;
    font-size: 0.95rem !important;
    max-height: 250px !important;
    overflow-y: auto !important;
    line-height: 1.7 !important;
    box-shadow: inset 5px 5px 15px rgba(0,0,0,0.3) !important;
}}

.final-prompt-container {{
    background: #FFFFFF !important; /* Clean White Background */
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    border-left: 6px solid #ff3b3b !important; /* Bold red accent bar */
    border-radius: 0 0 12px 12px !important;
    padding: 25px !important;
    font-family: 'Courier New', Courier, monospace !important;
    font-size: 0.9rem !important;
    color: #111111 !important; /* Crisp black text */
    max-height: 500px !important;
    overflow-y: auto !important;
    white-space: pre-wrap !important;
    line-height: 1.6 !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.06) !important; /* Soft card shadow */
}}

/* ── FINAL SYSTEM PROMPT SECTION ── */
.fp-section {{
    margin-top: 18px !important;
    margin-bottom: 6px !important;
}}

.fp-heading {{
    font-size: 0.72rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #CE1126 !important;
    margin-bottom: 8px !important;
    padding-left: 4px !important;
    border-left: 3px solid #FCD116 !important;
    padding-left: 8px !important;
}}

.fp-details {{
    border: 1.5px solid rgba(206,17,38,0.25) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    background: rgba(255,255,255,0.6) !important;
    transition: all 0.3s ease !important;
}}

.fp-details[open] {{
    border-color: rgba(206,17,38,0.5) !important;
    box-shadow: 0 6px 24px rgba(206,17,38,0.1) !important;
}}

.fp-summary {{
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    padding: 10px 16px !important;
    cursor: pointer !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #111111 !important;
    background: rgba(206,17,38,0.04) !important;
    user-select: none !important;
    list-style: none !important;
    transition: background 0.2s ease !important;
}}

.fp-summary::-webkit-details-marker {{ display: none !important; }}

.fp-summary:hover {{
    background: rgba(206,17,38,0.09) !important;
}}

.fp-chevron {{
    margin-left: auto !important;
    font-size: 0.75rem !important;
    color: #888 !important;
    transition: transform 0.3s ease !important;
    display: inline-block !important;
}}

.fp-details[open] .fp-chevron {{
    transform: rotate(180deg) !important;
}}

.rag-bubble-wrapper .stMarkdown h5 {{
    color: #CE1126 !important;
    font-weight: 800 !important;
    margin-bottom: 12px !important;
}}

/* ── PREMIUM GHANA-THEMED GRADIENT BACKGROUND ── */
.stApp::before {{
  content: '';
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background:
    radial-gradient(ellipse at 20% 20%, rgba(206, 17, 38, 0.08) 0%, transparent 50%),
    radial-gradient(ellipse at 80% 80%, rgba(0, 107, 63, 0.06) 0%, transparent 50%),
    radial-gradient(ellipse at 50% 50%, rgba(252, 209, 22, 0.04) 0%, transparent 60%);
  pointer-events: none;
  z-index: 0;
}}

/* ── CENTERED GHANA GRADIENT GLOW ── */
.stApp .main::before {{
  content: "";
  position: fixed;
  width: 600px;
  height: 600px;
  top: 40%;
  left: 50%;
  transform: translate(-50%, -50%);
  background: radial-gradient(circle,
    rgba(206, 17, 38, 0.10) 0%,
    rgba(252, 209, 22, 0.08) 40%,
    rgba(0, 107, 63, 0.06) 70%,
    transparent 100%
  );
  filter: blur(80px);
  z-index: 0;
  pointer-events: none;
}}

/* Subtle noise texture overlay */
.stApp::after {{
  content: '';
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E");
  opacity: 0.015;
  pointer-events: none;
  z-index: 0;
}}

*::-webkit-scrollbar {{ width:6px; height:6px; }}
*::-webkit-scrollbar-thumb {{ background:linear-gradient(180deg,#6f42ff,#00a2ff); border-radius:10px; }}
*::-webkit-scrollbar-track {{ background:transparent; }}

.chat-top-gradient {{
  position: fixed; top: 0 !important; left: 0; right: 0; width: 100vw;
  height: 12px; z-index: 99999 !important;
  background: linear-gradient(90deg, #CE1126, #FCD116, #006B3F);
}}

/* Fixed header container - independent of chat scroll */
.fixed-header-container {{
  position: fixed !important;
  top: 12px !important;
  left: 0 !important;
  right: 0 !important;
  z-index: 100000 !important;
  display: flex !important;
  align-items: center !important;
  gap: 8px !important;
  background: transparent !important;
  padding: 10px 24px !important;
  border-bottom: 1px solid var(--line) !important;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
  width: 100vw !important;
  box-sizing: border-box !important;
}}

/* Ensure the fixed header container's parent doesn't interfere */
.fixed-header-container > div {{
  position: static !important;
}}

/* Header button styles */
.header-btn {{
  height: 34px !important;
  width: 38px !important;
  border-radius: 8px !important;
  background: var(--surface) !important;
  border: 1px solid var(--line) !important;
  font-size: 18px !important;
  cursor: pointer !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  transition: all 0.2s ease !important;
  flex-shrink: 0 !important;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
}}

.header-btn:hover {{
  border-color: rgba(206,17,38,0.5) !important;
  transform: translateY(-2px) scale(1.05) !important;
  box-shadow: 0 4px 12px rgba(206,17,38,0.4) !important;
}}

.header-btn:active {{
  transform: translateY(0) scale(0.92) !important;
  box-shadow: 0 2px 8px rgba(206,17,38,0.5) !important;
}}

.fixed-header-container .stButton > button:hover {{
  transform: translateY(-2px) scale(1.05) !important;
  box-shadow: 0 4px 12px rgba(206,17,38,0.2) !important;
  border-color: rgba(206,17,38,0.3) !important;
}}

/* Scrollable chat container */
.chat-scrollable-container {{
  overflow-y: hidden !important;
  max-height: calc(100vh - 80px) !important;
  padding-top: 70px !important;
  padding-bottom: 120px !important;
  padding-left: 20px !important;
  padding-right: 20px !important;
}}

.chat-scrollable-container.has-scroll {{
  overflow-y: auto !important;
}}

/* ── HEADER BUTTONS STYLING ── */
button[kind="secondary"], button[data-testid="baseButton-secondary"] {{
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  height: 34px !important;
  width: 38px !important;
  padding: 0 !important;
  border-radius: 8px !important;
  background: var(--surface) !important;
  border: 1px solid var(--line) !important;
  box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
  cursor: pointer !important;
  transition: all 0.2s ease !important;
  font-size: 18px !important;
  line-height: 1 !important;
  min-width: 38px !important;
  min-height: 34px !important;
}}

button[kind="secondary"]:hover {{
  transform: translateY(-2px) scale(1.05) !important;
  box-shadow: 0 4px 12px rgba(206,17,38,0.2) !important;
  border-color: rgba(206,17,38,0.3) !important;
}}
.sys-status {{
  font-size: 0.75rem; color: var(--text-2); display: flex; align-items: center; justify-content: center; gap: 6px; font-weight: 600;
}}
.pulse-dot {{
  width: 8px; height: 8px; background: #10b981; border-radius: 50%;
  animation: pulse 2s infinite;
}}
@keyframes pulse {{
  0% {{ box-shadow: 0 0 0 0 rgba(16,185,129,0.7); }}
  70% {{ box-shadow: 0 0 0 6px rgba(16,185,129,0); }}
  100% {{ box-shadow: 0 0 0 0 rgba(16,185,129,0); }}
}}
.welcome-container {{
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 12px; padding: 24px; margin: 16px 1rem;
  box-shadow: 0 4px 16px rgba(0,0,0,0.05);
}}
.welcome-container h3 {{ margin-top: 0; color: var(--accent-1); }}
.sidebar-overlay {{
  position: fixed; top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.4); backdrop-filter: blur(2px);
  z-index: 999; display: none;
}}
.sidebar-overlay.open {{ display: block; }}

.custom-header-row {{
  display: flex !important; justify-content: flex-start !important; align-items: center !important; gap: 8px !important;
}}
.custom-header-row > div[data-testid="column"]:nth-child(1) {{
  flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; margin-right: 4px !important;
}}
.custom-header-row > div[data-testid="column"]:nth-child(2) {{
  flex: 0 0 auto !important; width: auto !important; min-width: 0 !important;
}}
.custom-header-row > div[data-testid="column"]:nth-child(3) {{
  flex: 0 0 auto !important; width: auto !important; min-width: 0 !important;
}}
.custom-header-row > div[data-testid="column"]:nth-child(4) {{
  flex: 1 1 auto !important; width: auto !important; min-width: 0 !important;
}}

.custom-sidebar-col {{
  height: 100vh;
  width: 100%;
  border-right: 1px solid var(--line);
}}

/* ── Buttons ── */
.stButton > button {{
  border-radius: 14px !important;
  border: 1.5px solid var(--line) !important;
  font-weight: 600 !important;
  font-family: 'Inter', sans-serif !important;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
  background: var(--surface) !important;
  color: var(--text-1) !important;
  padding: 0.5rem 1rem !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
}}
.stButton > button:hover {{
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 24px rgba(206, 17, 38, 0.12) !important;
  border-color: rgba(206, 17, 38, 0.3) !important;
}}
.stButton > button:active {{ transform: scale(0.97) !important; }}

/* ── Send button ── */
.send-btn .stButton > button {{
  background: linear-gradient(135deg, #6f42ff, #00a2ff) !important;
  color: white !important;
  border: none !important;
  border-radius: 14px !important;
  padding: 0.55rem 1.4rem !important;
  font-size: 0.95rem !important;
  box-shadow: 0 4px 18px rgba(111,66,255,0.35) !important;
  animation: none !important;
}}
.send-btn .stButton > button:hover {{
  box-shadow: 0 8px 28px rgba(111,66,255,0.5) !important;
  transform: translateY(-2px) scale(1.03) !important;
}}
.send-btn .stButton > button:active {{
  animation: btnBounce 0.35s ease !important;
}}
@keyframes btnBounce {{
  0%   {{ transform: scale(1); }}
  30%  {{ transform: scale(0.92); }}
  60%  {{ transform: scale(1.06); }}
  100% {{ transform: scale(1); }}
}}

/* ── Ghost icon buttons in header ── */
.icon-btn .stButton > button {{
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  border-radius: 10px !important;
  padding: 0.35rem 0.6rem !important;
  font-size: 1.1rem !important;
  color: var(--text-2) !important;
}}
.icon-btn .stButton > button:hover {{
  background: rgba(111,66,255,0.1) !important;
}}

/* ── New Chat button ── */
.new-chat-btn .stButton > button {{
  background: var(--surface) !important;
  color: #CE1126 !important;
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
  font-weight: 600 !important;
  transition: all 0.3s ease !important;
}}
.new-chat-btn .stButton > button:hover,
.new-chat-btn .stButton > button[kind="primary"]:hover,
.new-chat-btn .stButton > button[data-testid="baseButton-primary"]:hover {{
  transform: translateY(-2px) !important;
  border-color: #CE1126 !important;
  border: 2px solid #CE1126 !important;
  box-shadow: 0 4px 12px rgba(206,17,38,0.15) !important;
}}

/* ── Chat bubbles ── */
.bubble {{
  padding: 16px 20px; border-radius: 24px; margin: 8px 0;
  line-height: 1.7; max-width: 85%; word-wrap: break-word;
  white-space: pre-wrap; font-size: 0.95rem;
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}}
.msg-user {{ display:flex; justify-content:flex-end; padding: 0 1rem; }}
.msg-bot  {{ display:flex; justify-content:flex-start; padding: 0 1rem; }}
.bubble-user {{
  background: linear-gradient(135deg, #CE1126 0%, #8B0000 100%);
  color: white;
  box-shadow:
    0 4px 20px rgba(206, 17, 38, 0.25),
    0 2px 8px rgba(206, 17, 38, 0.15),
    inset 0 1px 0 rgba(255, 255, 255, 0.2);
  border-radius: 24px 24px 6px 24px;
  border: 1px solid rgba(255, 255, 255, 0.1);
}}
.bubble-user:hover {{
  transform: translateY(-2px);
  box-shadow:
    0 8px 30px rgba(206, 17, 38, 0.35),
    0 4px 12px rgba(206, 17, 38, 0.25),
    inset 0 1px 0 rgba(255, 255, 255, 0.3);
}}
.bubble-bot {{
  background: rgba(255, 255, 255, 0.95);
  color: var(--text-1);
  border: 1px solid rgba(206, 17, 38, 0.08);
  box-shadow:
    0 4px 24px rgba(0, 0, 0, 0.08),
    0 2px 8px rgba(0, 0, 0, 0.04);
  border-radius: 24px 24px 24px 6px;
}}
.bubble-bot:hover {{
  transform: translateY(-1px);
  box-shadow:
    0 6px 32px rgba(0, 0, 0, 0.12),
    0 3px 12px rgba(0, 0, 0, 0.06);
}}
.bot-name {{
  font-size: 0.75rem; font-weight: 700; color: #CE1126;
  margin-bottom: 8px; display:flex; align-items:center; gap:6px;
  letter-spacing: 0.02em;
}}
.intent-badge {{
  font-size: 0.65rem; padding: 4px 12px; border-radius: 999px;
  font-weight: 700; letter-spacing: 0.05em;
  text-transform: uppercase;
  background: rgba(255, 255, 255, 0.95);
  border: 1.5px solid rgba(0, 0, 0, 0.1);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}}
.badge-election {{
  background: #fff5f5;
  color: #CE1126;
  border: 1.5px solid #CE1126;
}}
.badge-budget {{
  background: #f0fdf4;
  color: #006B3F;
  border: 1.5px solid #006B3F;
}}
.badge-compare {{
  background: #fefce8;
  color: #a37f00;
  border: 1.5px solid #a37f00;
}}
.msg-actions {{
  display: flex;
  gap: 8px;
  justify-content: flex-end;
  margin-top: -4px;
  margin-bottom: 8px;
  padding-right: 1.2rem;
  opacity: 0.6;
  transition: opacity 0.2s;
}}
.msg-actions:hover {{ opacity: 1; }}
.action-btn {{
  cursor: pointer;
  font-size: 0.85rem;
  background: transparent;
  border: none;
  padding: 2px 5px;
  border-radius: 4px;
  color: var(--text-2);
}}
.action-btn:hover {{
  background: rgba(0,0,0,0.05);
  color: var(--red-gh);
}}
/* ── ULTRA-AGGRESSIVE EMOJI BUTTON RESET ── */
/* This targets the Streamlit block containing our marker div */
div:has(> .user-msg-actions) button {{
  background: transparent !important;
  background-color: transparent !important;
  background-repeat: no-repeat !important;
  background-position: center !important;
  background-size: 14px 14px !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  min-height: 0 !important;
  height: 22px !important;
  width: 22px !important;
  margin: 0 !important;
  border-radius: 4px !important;
  color: transparent !important; /* Hide the text/emoji */
  transition: all 0.2s ease !important;
}}

/* Line sketch icon for Copy - Matching screenshot style */
div:has(> .user-msg-actions) [data-testid="column"]:nth-of-type(2) button {{
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='9' y='9' width='13' height='13' rx='3' ry='3'%3E%3C/rect%3E%3Cpath d='M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1'%3E%3C/path%3E%3C/svg%3E") !important;
  background-size: 16px 16px !important;
}}

/* Line sketch icon for Edit - Matching screenshot style */
div:has(> .user-msg-actions) [data-testid="column"]:nth-of-type(3) button {{
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M11 20h10'%3E%3C/path%3E%3Cpath d='M15.5 4.5a1.5 1.5 0 0 1 2 2L6 18l-3 1 1-3 11.5-11.5z'%3E%3C/path%3E%3C/svg%3E") !important;
  background-size: 16px 16px !important;
}}

div:has(> .user-msg-actions) button:hover {{
  background-color: rgba(206,17,38,0.08) !important;
  transform: scale(1.1) !important;
}}

/* Red hover state for icons */
div:has(> .user-msg-actions) [data-testid="column"]:nth-of-type(2) button:hover {{
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23CE1126' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='9' y='9' width='13' height='13' rx='3' ry='3'%3E%3C/rect%3E%3Cpath d='M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1'%3E%3C/path%3E%3C/svg%3E") !important;
}}
div:has(> .user-msg-actions) [data-testid="column"]:nth-of-type(3) button:hover {{
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23CE1126' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M11 20h10'%3E%3C/path%3E%3Cpath d='M15.5 4.5a1.5 1.5 0 0 1 2 2L6 18l-3 1 1-3 11.5-11.5z'%3E%3C/path%3E%3C/svg%3E") !important;
}}

div:has(> .user-msg-actions) [data-testid="stHorizontalBlock"] {{
  background: transparent !important;
  margin-top: -12px !important;
  gap: 2px !important;
}}

/* ── TYPING INDICATOR ANIMATION ── */
.typing-indicator {{
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 18px 24px;
  margin: 8px 0;
  background: rgba(255, 255, 255, 0.95);
  border-radius: 24px 24px 24px 6px;
  box-shadow:
    0 4px 20px rgba(0, 0, 0, 0.08),
    0 2px 8px rgba(0, 0, 0, 0.04);
  border: 1px solid rgba(206, 17, 38, 0.08);
}}

.typing-indicator .dot {{
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: linear-gradient(135deg, #CE1126, #8B0000);
  animation: typingBounce 1.4s infinite ease-in-out both;
  box-shadow: 0 2px 8px rgba(206, 17, 38, 0.3);
}}

.typing-indicator .dot:nth-child(1) {{ animation-delay: -0.32s; }}
.typing-indicator .dot:nth-child(2) {{ animation-delay: -0.16s; }}
.typing-indicator .dot:nth-child(3) {{ animation-delay: 0s; }}

@keyframes typingBounce {{
  0%, 80%, 100% {{ transform: scale(0.5); opacity: 0.3; }}
  40% {{ transform: scale(1.1); opacity: 1; }}
}}

/* ── PREMIUM EMPTY STATE ── */
.empty-state {{
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  padding: 20px 20px 60px;
  text-align: center;
}}

/* Slide up animation for empty state elements */
@keyframes slideUp {{
  from {{ opacity: 0; transform: translateY(60px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}

/* Animated gradient title - now applied to main title */
.main-title {{
  font-size: 42px;
  font-weight: 800;
  letter-spacing: -0.03em;
  background: linear-gradient(135deg, #CE1126 0%, #F4A261 50%, #006B3F 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 12px;
  animation: slideUp 0.6s ease-out forwards, float 3s ease-in-out infinite 0.6s;
  filter: drop-shadow(0 4px 20px rgba(206,17,38,0.2));
  opacity: 0;
}}

@keyframes float {{
  0% {{ transform: translateY(0px); }}
  50% {{ transform: translateY(-8px); }}
  100% {{ transform: translateY(0px); }}
}}

/* Subtitle with staggered animation */
.sub-text {{
  font-size: 15px;
  color: var(--text-2);
  font-weight: 400;
  animation: slideUp 0.6s ease-out 0.15s forwards;
  opacity: 0;
  margin-bottom: 10px;
}}

/* Containers wrapper slide up */
.insight-containers-wrapper {{
  animation: slideUp 0.6s ease-out 0.3s forwards;
  opacity: 0;
}}

/* ── SUGGESTION CHIPS ── */
.suggestion-chips {{
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 16px;
  padding: 0 20px 40px;
  flex-wrap: nowrap;
  animation: fadeUp 0.8s ease 0.6s forwards;
  opacity: 0;
  max-width: 900px;
  margin: 0 auto;
}}

@media (max-width: 768px) {{
  .suggestion-chips {{
    flex-wrap: wrap;
    gap: 12px;
  }}
}}

.suggestion-chips > div {{
  flex: 1;
  min-width: 200px;
  max-width: 280px;
}}

.suggestion-chips button {{
  background: rgba(255,255,255,0.7) !important;
  border: 1px solid rgba(255,255,255,0.3) !important;
  border-radius: 20px !important;
  padding: 16px 24px !important;
  font-size: 0.9rem !important;
  font-weight: 500 !important;
  color: var(--text-1) !important;
  box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
  backdrop-filter: blur(10px) !important;
  -webkit-backdrop-filter: blur(10px) !important;
  transition: all 0.25s ease !important;
  cursor: pointer !important;
  width: 100% !important;
  min-height: 60px !important;
}}

.suggestion-chips button:hover {{
  transform: translateY(-6px) scale(1.02) !important;
  box-shadow: 0 10px 25px rgba(0,0,0,0.08) !important;
  background: rgba(255,255,255,0.9) !important;
  border-color: rgba(206,17,38,0.2) !important;
}}

/* ── HORIZONTAL SUGGESTION CARDS ── */
.suggestion-cards-wrapper {{
  display: flex;
  flex-direction: row;
  justify-content: center;
  align-items: stretch;
  gap: 20px;
  padding: 20px;
  max-width: 1000px;
  margin: 0 auto;
}}

/* ── 3 HORIZONTAL INSIGHT CONTAINERS ── */
.insight-containers-wrapper {{
  display: flex;
  flex-direction: row;
  justify-content: center;
  align-items: stretch;
  flex-wrap: wrap;
  gap: 16px;
  padding: 20px 40px;
  max-width: 900px;
  margin: 20px auto 0;
}}

.insight-container {{
  flex: 1 1 200px;
  min-width: 200px;
  max-width: 280px;
  background: rgba(255,255,255,0.7);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-radius: 16px;
  padding: 20px;
  text-align: center;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
  word-wrap: normal;
  overflow-wrap: normal;
}}

/* Animated Ghanaian gradient borders - each edge moves independently */
.insight-container .border-top {{
  position: absolute;
  top: 0;
  left: 0;
  height: 3px;
  background: linear-gradient(90deg, transparent, #CE1126, #FCD116, #006B3F);
  animation: borderTop 3s linear infinite;
  z-index: 2;
  border-radius: 2px;
}}

.insight-container .border-right {{
  position: absolute;
  top: 0;
  right: 0;
  width: 3px;
  background: linear-gradient(180deg, transparent, #CE1126, #FCD116, #006B3F);
  animation: borderRight 3s linear infinite;
  z-index: 2;
  border-radius: 2px;
}}

.insight-container .border-bottom {{
  position: absolute;
  bottom: 0;
  right: 0;
  height: 3px;
  background: linear-gradient(270deg, transparent, #CE1126, #FCD116, #006B3F);
  animation: borderBottom 3s linear infinite;
  z-index: 2;
  border-radius: 2px;
}}

.insight-container .border-left {{
  position: absolute;
  bottom: 0;
  left: 0;
  width: 3px;
  background: linear-gradient(0deg, transparent, #CE1126, #FCD116, #006B3F);
  animation: borderLeft 3s linear infinite;
  z-index: 2;
  border-radius: 2px;
}}

@keyframes borderTop {{
  0% {{ width: 0; left: 0; opacity: 0; }}
  10% {{ opacity: 1; }}
  40% {{ width: 100%; left: 0; opacity: 1; }}
  50% {{ width: 0; left: 100%; opacity: 0; }}
  100% {{ width: 0; left: 0; opacity: 0; }}
}}

@keyframes borderRight {{
  0% {{ height: 0; top: 0; opacity: 0; }}
  12.5% {{ height: 0; top: 0; opacity: 0; }}
  22.5% {{ opacity: 1; }}
  50% {{ height: 100%; top: 0; opacity: 1; }}
  62.5% {{ height: 0; top: 100%; opacity: 0; }}
  100% {{ height: 0; top: 0; opacity: 0; }}
}}

@keyframes borderBottom {{
  0% {{ width: 0; right: 0; opacity: 0; }}
  25% {{ width: 0; right: 0; opacity: 0; }}
  35% {{ opacity: 1; }}
  62.5% {{ width: 100%; right: 0; opacity: 1; }}
  75% {{ width: 0; right: 100%; opacity: 0; }}
  100% {{ width: 0; right: 0; opacity: 0; }}
}}

@keyframes borderLeft {{
  0% {{ height: 0; bottom: 0; opacity: 0; }}
  37.5% {{ height: 0; bottom: 0; opacity: 0; }}
  47.5% {{ opacity: 1; }}
  75% {{ height: 100%; bottom: 0; opacity: 1; }}
  87.5% {{ height: 0; bottom: 100%; opacity: 0; }}
  100% {{ height: 0; bottom: 0; opacity: 0; }}
}}

.insight-container:hover .border-top,
.insight-container:hover .border-right,
.insight-container:hover .border-bottom,
.insight-container:hover .border-left {{
  animation-duration: 1.5s;
}}

.insight-container:hover {{
  transform: translateY(-4px);
  box-shadow: 0 12px 30px rgba(0,0,0,0.1);
}}

.insight-container-title {{
  font-size: 14px;
  font-weight: 700;
  color: var(--text-1);
  margin-bottom: 8px;
  word-wrap: normal;
  overflow-wrap: normal;
  white-space: normal;
}}

.insight-container-desc {{
  font-size: 12px;
  font-weight: 400;
  color: var(--text-2);
  line-height: 1.5;
  word-wrap: normal;
  overflow-wrap: normal;
  white-space: normal;
}}

.suggestion-card {{
  flex: 1;
  min-width: 200px;
  max-width: 300px;
}}

.suggestion-card button {{
  background: rgba(255,255,255,0.75) !important;
  border: 1px solid rgba(255,255,255,0.4) !important;
  border-radius: 16px !important;
  padding: 18px 24px !important;
  font-size: 0.9rem !important;
  font-weight: 500 !important;
  color: var(--text-1) !important;
  box-shadow: 0 4px 12px rgba(0,0,0,0.06) !important;
  backdrop-filter: blur(12px) !important;
  -webkit-backdrop-filter: blur(12px) !important;
  transition: all 0.3s ease !important;
  cursor: pointer !important;
  width: 100% !important;
  min-height: 64px !important;
}}

.suggestion-card button:hover {{
  transform: translateY(-4px) scale(1.01) !important;
  box-shadow: 0 12px 30px rgba(0,0,0,0.1) !important;
  background: rgba(255,255,255,0.95) !important;
  border-color: rgba(206,17,38,0.3) !important;
}}

/* ── Glassmorphism utility class ── */
.glass-card {{
  background: rgba(255,255,255,0.6) !important;
  backdrop-filter: blur(12px) !important;
  -webkit-backdrop-filter: blur(12px) !important;
  border-radius: 20px !important;
  border: 1px solid rgba(255,255,255,0.3) !important;
  box-shadow: 0 8px 32px rgba(0,0,0,0.08) !important;
}}

/* ── Response cards ── */
.resp-card {{
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 14px 18px;
  margin-top: 8px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.06);
}}
.resp-card-election {{ border-left: 4px solid var(--red-gh); }}
.resp-card-budget   {{ border-left: 4px solid var(--green-gh); }}
.resp-card-compare  {{ border-left: 4px solid var(--gold-gh); }}
.source-chip {{
  display: inline-flex; align-items: center; gap: 4px;
  background: rgba(111,66,255,0.08); border: 1px solid rgba(111,66,255,0.2);
  border-radius: 8px; padding: 2px 10px; font-size: 0.74rem;
  font-weight: 600; color: var(--accent-1); margin: 3px 2px; cursor: default;
}}

/* ── Sidebar ── */
.sidebar-wrap {{
  display: flex;
  flex-direction: column;
}}
.sidebar-content {{
  flex: 1;
  overflow-y: auto;
}}
.sidebar-brand {{
  display: flex; align-items: center; gap: 8px;
  font-weight: 800; font-size: 1.05rem; color: var(--red-gh);
  padding-bottom: 12px;
  border-bottom: 1px solid var(--line);
  margin-bottom: 10px;
}}
.sidebar-label {{
  font-size: 0.65rem; font-weight: 700; color: var(--text-2);
  text-transform: uppercase; letter-spacing: 0.08em;
  margin: 12px 0 5px;
}}
.settings-card {{
  background: var(--surface) !important;
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 12px 10px 12px;
  margin-bottom: 4px;
  overflow: visible !important;
}}
.group-label {{
  font-size: 0.65rem; font-weight: 700; color: var(--text-2);
  text-transform: uppercase; letter-spacing: 0.06em;
  margin: 10px 2px 4px;
}}

/* ── Sidebar Chat History Buttons ── */
[data-testid="stVerticalBlock"] .stButton > button {{
  width: 100% !important;
  min-width: 0 !important;
  max-width: 100% !important;
}}

/* Sidebar chat buttons - red hover border */
[data-testid="stVerticalBlock"] button[kind="primary"]:hover,
[data-testid="stVerticalBlock"] button[kind="secondary"]:hover {{
  border-color: #CE1126 !important;
  border: 2px solid #CE1126 !important;
}}
.pin-dot {{
  width: 6px; height: 6px; border-radius: 50%;
  background: #FCD116; display: inline-block; margin-right: 4px;
}}

/* ── Chat header ── */
.chat-header-bar {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 0 10px;
  border-bottom: 1px solid var(--line);
  margin-bottom: 12px;
}}
.header-left {{
  display: flex; align-items: center; gap: 4px;
}}
.header-brand {{
  display: flex;
  align-items: center;
  gap: 6px;
  font-weight: 900;
  font-size: 1.1rem;
  color: #CE1126;
  margin: 0 !important;
  padding: 0 !important;
}}
.header-flex-marker + div {{
  margin-top: 0 !important;
  padding-top: 4px !important;
}}

/* ── Input area ── */
.input-inner {{
  max-width: 560px;
  margin: 0 auto;
  display: flex;
  align-items: flex-end;
  gap: 10px;
  padding: 0 16px;
}}




/* ── Expanders ── */
[data-testid="stExpander"] {{
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
  background: var(--surface) !important;
}}

/* ── Cards ── */
.stat-card {{
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 18px 22px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.05);
  margin-bottom: 12px;
}}

/* ── Ghana landing ── */
@keyframes ghanaGrad {{
  0%   {{ background-position: 0% 50%; }}
  50%  {{ background-position: 100% 50%; }}
  100% {{ background-position: 0% 50%; }}
}}
.gh-landing {{
  position: fixed; inset: 0;
  background: linear-gradient(-45deg,
    #006B3F, #004d2d, #CE1126, #8b0000,
    #FCD116, #b8960a, #006B3F, #CE1126);
  background-size: 500% 500%;
  animation: ghanaGrad 12s ease infinite;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  z-index: 9999;
}}
.gh-glass {{
  background: rgba(255,255,255,0.08);
  backdrop-filter: blur(24px) saturate(1.4);
  -webkit-backdrop-filter: blur(24px) saturate(1.4);
  border: 1px solid rgba(255,255,255,0.22);
  border-radius: 28px;
  padding: 48px 54px 40px;
  text-align: center;
  max-width: 640px;
  width: 90vw;
  box-shadow: 0 24px 60px rgba(0,0,0,0.35);
}}
.gh-logo-bar {{
  position: fixed; top: 22px; left: 26px;
  display: flex; align-items: center; gap: 9px;
  z-index: 10000;
}}
.gh-logo-bar img {{ width: 24px; height: 24px; object-fit: contain; }}
.gh-logo-bar span {{
  font-size: 18px; font-weight: 800; color: #fff;
  text-shadow: 0 2px 8px rgba(0,0,0,0.4);
}}
.gh-title {{
  font-size: clamp(48px, 8vw, 82px);
  font-weight: 900; letter-spacing: -2px;
  color: #ffffff;
  text-shadow: 0 4px 24px rgba(0,0,0,0.4);
  line-height: 1; margin-bottom: 14px;
}}
.gh-subtitle {{
  color: rgba(255,255,255,0.85);
  font-size: clamp(15px, 1.6vw, 19px);
  margin-bottom: 6px; font-weight: 500;
}}
.typing-container {{
  height: 2.2em; position: relative;
  overflow: hidden; margin: 8px 0 28px;
}}
.typing-phrase {{
  position: absolute; left: 0; right: 0;
  color: #FCD116;
  font-size: clamp(14px, 1.4vw, 17px);
  font-weight: 600; opacity: 0;
  animation: phraseAnim 15s infinite;
  text-shadow: 0 2px 8px rgba(0,0,0,0.3);
}}
.typing-phrase:nth-child(1) {{ animation-delay: 0s; }}
.typing-phrase:nth-child(2) {{ animation-delay: 5s; }}
.typing-phrase:nth-child(3) {{ animation-delay: 10s; }}
@keyframes phraseAnim {{
  0%   {{ opacity: 0; transform: translateY(8px); }}
  8%   {{ opacity: 1; transform: translateY(0); }}
  28%  {{ opacity: 1; transform: translateY(0); }}
  35%  {{ opacity: 0; transform: translateY(-8px); }}
  100% {{ opacity: 0; }}
}}
/* Start Now button — fixed below the glass card */
.st-key-start_now_btn {{
  position: fixed !important;
  left: 50% !important;
  top: calc(50% + 220px) !important;
  transform: translateX(-50%) !important;
  z-index: 10001 !important;
  width: min(260px, 44vw) !important;
}}
.st-key-start_now_btn .stButton > button {{
  background: linear-gradient(135deg, #FCD116, #f0a500) !important;
  color: #111 !important; border: none !important;
  font-size: 1.05rem !important; font-weight: 700 !important;
  padding: 0.78rem 0 !important;
  width: 100% !important;
  border-radius: 999px !important;
  box-shadow: 0 8px 28px rgba(252,209,22,0.55) !important;
  letter-spacing: 0.02em !important;
}}
.st-key-start_now_btn .stButton > button:hover {{
  transform: translateY(-2px) scale(1.03) !important;
  box-shadow: 0 14px 36px rgba(252,209,22,0.65) !important;
}}

/* ── Stage logs ── */
.stage-row {{ display:flex; gap:8px; align-items:center; padding:5px 0; border-bottom:1px solid var(--line); }}
.stage-pill {{ background:linear-gradient(90deg,#6f42ff,#00a2ff); color:white; border-radius:8px; padding:2px 9px; font-size:0.72rem; font-weight:700; white-space:nowrap; }}
.stage-info {{ color:var(--text-2); font-size:0.8rem; }}

/* RAG Details expander scrolling */
[data-testid="stExpander"] {{
  max-height: 400px !important;
  overflow-y: auto !important;
}}
[data-testid="stExpander"] > div {{
  max-height: 400px !important;
  overflow-y: auto !important;
}}

/* ── FINAL STREAMLIT TOP GAP KILLER (CLEANED) ── */
.stApp {{
  margin-top: 0 !important;
  padding-top: 0 !important;
  background: transparent !important;
}}

[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"],
.stMain,
main {{
  padding-top: 0 !important;
  margin-top: 0 !important;
}}

/* force first element to start at absolute top */
.element-container:first-child {{
  margin-top: 0 !important;
  padding-top: 0 !important;
}}

/* ── HARD RESET STREAMLIT TOP SPACING ── */
html, body, .stApp {{
  margin: 0 !important;
  padding: 0 !important;
  overflow-x: hidden !important;
  /* Legacy solid color override removed */
}}

/* Kill Streamlit header space completely */
[data-testid="stHeader"] {{
  display: none !important;
  height: 0px !important;
}}

/* Main container reset */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"] {{
  padding-top: 0 !important;
  margin-top: 0 !important;
}}

/* Responsive: Adjust for smaller screens */
@media (max-width: 768px) {{
  div[data-testid="stChatInput"] {{
    min-width: 300px !important;
  }}
}}

</style>
""", unsafe_allow_html=True)


# ── Landing page ───────────────────────────────────────────────────────────────

def render_landing(logo_html):
    # Full-page Ghana gradient with glassy card
    st.markdown(f"""
    <div class="gh-logo-bar">
      {logo_html}
      <span>acitypal</span>
    </div>
    <div class="gh-landing">
      <div class="gh-glass">
        <div class="gh-title">ACITYPAL</div>
        <div class="gh-subtitle">Your Academic City Intelligent Assistant<br>Ghana Elections &amp; 2025 Budget</div>
        <div class="typing-container">
          <div class="typing-phrase">❝ Who won Ashanti Region in 2020? ❞</div>
          <div class="typing-phrase">❝ What is the 2025 budget deficit target? ❞</div>
          <div class="typing-phrase">❝ Compare NPP vs NDC votes in Greater Accra. ❞</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Start Now button
    if st.button("Start Now", key="start_now_btn"):
        st.session_state.started = True
        st.session_state.has_chart = False
        if st.session_state.active_chat_idx is None:
            create_new_chat()
        st.rerun()

    st.stop()


# ── Sidebar ────────────────────────────────────────────────────────────────────

def _group_conversations(convs):
    """Group conversations by Today / Yesterday / This Week / Older."""
    today = datetime.now().date()
    groups = {"📌 Pinned": [], "🕐 Today": [], "📅 Yesterday": [],
              "🗓 This Week": [], "📁 Older": []}
    for i, c in enumerate(convs):
        if c.get("pinned"):
            groups["📌 Pinned"].append(i)
            continue
        try:
            d = datetime.strptime(c["created_at"], "%Y-%m-%d %H:%M").date()
        except Exception:
            d = today
        delta = (today - d).days
        if delta == 0:
            groups["🕐 Today"].append(i)
        elif delta == 1:
            groups["📅 Yesterday"].append(i)
        elif delta <= 7:
            groups["🗓 This Week"].append(i)
        else:
            groups["📁 Older"].append(i)
    return groups


def _export_chat_markdown(chat) -> str:
    lines = [f"# {chat.get('title','Chat')}", f"_Exported from AcityPal — {datetime.now().strftime('%Y-%m-%d %H:%M')}_", ""]
    for m in chat.get("messages", []):
        role = "**You**" if m["role"] == "user" else "**AcityPal**"
        lines.append(f"{role}: {m['content']}")
        lines.append("")
    return "\n".join(lines)


def render_sidebar(logo_html, pipeline):
    # Wrap everything in a padded container to ensure space from the red border
    st.markdown('<div class="sidebar-internal-content">', unsafe_allow_html=True)
       
    # ── Brand + dark mode + close (all on same horizontal line) ─────────────────
    col1, col2, col3 = st.columns([0.7, 0.15, 0.15], vertical_alignment="center")
    
    with col1:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-left: -5px;">
            {logo_html}
            <div style="display: flex; flex-direction: column; line-height: 1.1;">
                <span style="color:#CE1126;font-weight:800;font-size:15px;letter-spacing:-0.01em;">acitypal</span>
                <span style="color:var(--text-2);font-weight:600;font-size:9px;opacity:0.7;">Smart civic insights</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Dark mode toggle - emoji only in red
        theme_emoji = "🌙" if st.session_state.theme_mode == "light" else "☀️"
        if st.button(theme_emoji, key="sb_theme_emoji", help="Toggle dark/light mode"):
            st.session_state.theme_mode = "dark" if st.session_state.theme_mode == "light" else "light"
            st.rerun()
    
    with col3:
        # X button with tooltip
        if st.button("✕", key="sb_close", help="Close Sidebar"):
            st.session_state.show_sidebar = False
            st.rerun()

    # ── New Chat ───────────────────────────────────────────────────────────────
    new_chat_container = st.container()
    with new_chat_container:
        st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
        if st.button("➕ New Chat", key="sidebar_new_chat", use_container_width=True, type="primary"):
            create_new_chat()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Retrieval Settings ─────────────────────────────────────────────────────
    st.session_state.top_k = st.slider(
        f"Top-K Chunks  `{st.session_state.top_k}`",
        1, 10, st.session_state.top_k, key="topk_sl",
        help="How many context chunks to retrieve per query"
    )
    st.session_state.use_hybrid = st.toggle(
        "Hybrid Search", value=st.session_state.use_hybrid, key="hyb_tog",
        help="Blends vector similarity (75%) + TF-IDF keyword (25%)"
    )
    st.session_state.prompt_variant_sel = st.selectbox(
        "Prompt Style",
        ["strict", "base", "chain_of_thought", "hybrid"],
        index=["strict","base","chain_of_thought", "hybrid"].index(st.session_state.get("prompt_variant_sel","hybrid")),
        key="pv_sel",
        help="strict = grounded rules | base = minimal | chain_of_thought = step-by-step reasoning | hybrid = intent-aware rules"
    )
    st.session_state.region_filter = st.selectbox(
        "Region Filter",
        GHANA_REGIONS,
        index=GHANA_REGIONS.index(st.session_state.get("region_filter","All Regions")),
        key="reg_sel",
        help="Restrict election retrieval to a specific Ghana region"
    )

    # ── Chat History (grouped) ─────────────────────────────────────────────────

    if not st.session_state.conversations:
        st.caption("No conversations yet.")
    else:
        groups = _group_conversations(st.session_state.conversations)
        active_idx = st.session_state.active_chat_idx
        for group_name, indices in groups.items():
            if not indices:
                continue
            st.markdown(f'<div class="group-label">{group_name}</div>', unsafe_allow_html=True)
            for idx in indices:
                conv = st.session_state.conversations[idx]
                # Smart title: Use existing title or first user message
                title = conv.get("title")
                if not title or title == "New Chat":
                    user_msgs = [m["content"] for m in conv.get("messages", []) if m["role"] == "user"]
                    title = user_msgs[0] if user_msgs else f"Chat {idx+1}"
                
                is_pinned = conv.get("pinned", False)
                is_active = idx == active_idx
                pin_label = "📍 Unpin" if is_pinned else "📌 Pin"

                r1, r3 = st.columns([0.86, 0.14], vertical_alignment="center")
                with r1:
                    label = f"{title[:22]}{'…' if len(title)>22 else ''}"
                    btn_type = "primary" if is_active else "secondary"
                    if st.button(label, key=f"sc_{idx}", use_container_width=True, type=btn_type):
                        st.session_state.active_chat_idx = idx
                        st.session_state.show_sidebar = False
                        st.rerun()
                with r3:
                    # Use a simple button with a container that appears below
                    menu_key = f"menu_open_{idx}"
                    
                    if st.button("⋮", key=f"three_dots_{idx}", help="Options"):
                        st.session_state[menu_key] = not st.session_state.get(menu_key, False)
                        st.rerun()
                    
                    # Show menu below if open
                    if st.session_state.get(menu_key, False):
                        # Close menu when clicking outside (by clicking the same button)
                        st.markdown(f"""
                        <div style="position: absolute; background: var(--surface); border: 1px solid var(--line); 
                                    border-radius: 8px; padding: 8px; z-index: 10000; 
                                    width: 120px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                            <div style="padding: 6px 10px; cursor: pointer;" onclick="this.style.backgroundColor='rgba(0,0,0,0.05)'">
                                ✏️ Rename
                            </div>
                            <div style="padding: 6px 10px; cursor: pointer;" 
                                 onclick="document.querySelector('[data-testid=\'stMarkdown\']').click()">
                                {pin_label}
                            </div>
                            <div style="padding: 6px 10px; cursor: pointer;">📤 Share</div>
                            <hr style="margin: 4px 0;">
                            <div style="padding: 6px 10px; cursor: pointer; color: #CE1126;">🗑 Delete</div>
                        </div>
                        """, unsafe_allow_html=True)
    # Close the internal content div
    st.markdown('</div>', unsafe_allow_html=True)


# ── Response rendering ─────────────────────────────────────────────────────────

def render_vote_chart(vote_data: list):
    """Render a Chart.js bar chart for vote data."""
    if not vote_data:
        return
    st.session_state.has_chart = True
    labels = json.dumps([d["label"] for d in vote_data])
    values = json.dumps([d["votes"] for d in vote_data])
    colors = json.dumps(["#CE1126","#006B3F","#FCD116","#4f46e5","#0ea5e9",
                          "#f97316","#8b5cf6","#10b981"][:len(vote_data)])
    chart_html = f"""
<!DOCTYPE html><html><body style="margin:0;padding:0;background:transparent;">
<canvas id="vc" style="max-height:220px;"></canvas>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
new Chart(document.getElementById('vc'), {{
  type: 'bar',
  data: {{
    labels: {labels},
    datasets: [{{
      data: {values},
      backgroundColor: {colors},
      borderRadius: 8,
      borderSkipped: false,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        callbacks: {{
          label: ctx => ' ' + ctx.raw.toLocaleString() + ' votes'
        }}
      }}
    }},
    scales: {{
      y: {{
        beginAtZero: true,
        ticks: {{ callback: v => (v/1000).toFixed(0)+'K' }},
        grid: {{ color: 'rgba(200,200,200,0.15)' }}
      }},
      x: {{ grid: {{ display: false }} }}
    }}
  }}
}});
</script>
</body></html>"""
    st.html(chart_html)


def render_source_chips(retrieved_docs: list):
    """Render clickable source citation chips."""
    chips = []
    seen = set()
    for d in retrieved_docs:
        md = d.get("metadata", {})
        src = md.get("source", "")
        if src == "csv":
            tag = f"Election CSV · {md.get('region','?')} {md.get('year','')}"
        else:
            tag = f"2025 Budget · Page {md.get('page','?')}"
        score = d.get("score", 0)
        key = tag
        if key not in seen:
            seen.add(key)
            chips.append(f'<span class="source-chip" title="Score: {score:.4f}">{tag}</span>')
    if chips:
        st.markdown(
            '<div style="margin-top:8px;">' + "".join(chips) + '</div>',
            unsafe_allow_html=True,
        )


# ── Chat area ──────────────────────────────────────────────────────────────────

def render_chat_tab(pipeline, logo_html):
    if st.session_state.active_chat_idx is None or \
       st.session_state.active_chat_idx >= len(st.session_state.conversations):
        create_new_chat()

    chat = get_active_chat()

    if st.session_state.show_sidebar:
        col_s, col_c = st.columns([1.3, 2.7], gap="small")  # widened sidebar ratio

        with col_s:
            # Marker and Physical Edge Div
            st.markdown('<div class="sidebar-panel-fix"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-panel-fix-edge"></div>', unsafe_allow_html=True)
            
            # ── PERSISTENT JAVASCRIPT: FORCE CENTERING ONLY ──
            st.markdown("""
            <script>
                const forceCenter = () => {
                    const marker = window.parent.document.querySelector('.sidebar-panel-fix');
                    if (marker) {
                        const col = marker.closest('[data-testid="column"]');
                        if (col) {
                            const widgets = col.querySelectorAll('[data-testid="stVerticalBlock"] > div, [data-testid="stHorizontalBlock"]');
                            widgets.forEach(w => {
                                w.style.paddingLeft = '80px';
                                w.style.paddingRight = '80px';
                                w.style.boxSizing = 'border-box';
                            });
                        }
                    }
                };
                setInterval(forceCenter, 500);
            </script>
            """, unsafe_allow_html=True)
            
            render_sidebar(logo_html, pipeline)

        with col_c:
            with st.container():
                st.markdown('<div class="chat-panel-fix"></div>', unsafe_allow_html=True)
                _render_chat_area(chat, pipeline, logo_html)

    else:
        _render_chat_area(chat, pipeline, logo_html)


def render_fixed_header(logo_html):
    """Render fixed header that appears when sidebar is closed."""
    # Only show header when sidebar is closed
    if not st.session_state.get("show_sidebar", False):
        # Create a container with buttons and header
        st.markdown("""
        <div style="position: fixed; top: 12px; left: 0; right: 0; z-index: 100000; display: flex; align-items: center; gap: 20px; background: transparent; padding: 10px 24px;">
        """, unsafe_allow_html=True)

        col_logo, col_burger, col_new, col_spacer = st.columns([0.8, 0.4, 0.04, 8.28])

        with col_logo:
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 0px;">
                {logo_html}
                <div style="display: flex; flex-direction: column; line-height: 1.1;">
                    <span style="color:#CE1126;font-weight:800;font-size:16px;letter-spacing:-0.02em;">acitypal</span>
                    <span style="color:var(--text-2);font-weight:500;font-size:10px;opacity:0.75;">Smart civic insights</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_burger:
            if st.button("☰", key="header_burger_fixed", help="Open Sidebar"):
                st.session_state.show_sidebar = True
                st.rerun()

        with col_new:
            if st.button("＋", key="header_new_fixed", help="New Chat"):
                create_new_chat()
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def _render_chat_area(chat, pipeline, logo_html=""):
    st.markdown('<div class="chat-top-gradient"></div>', unsafe_allow_html=True)

    # CSS for scrollable container - enable scrolling with auto-scroll
    has_msgs = len(chat["messages"]) > 0
    st.markdown(f"""
    <style>
    .chat-scrollable-container {{
      overflow-y: auto !important;
      max-height: calc(100vh - 80px) !important;
      padding-top: 20px !important;
      padding-bottom: 120px !important;
      padding-left: 20px !important;
      padding-right: 20px !important;
      scroll-behavior: smooth !important;
      scroll-snap-type: y proximity !important;
    }}
    
    /* Auto-scroll to bottom using CSS */
    .chat-scrollable-container::after {{
      content: "";
      display: block;
      height: 1px;
      scroll-snap-align: end;
      margin-bottom: -1px;
    }}
    
    /* Force scroll to bottom on render */
    @keyframes scrollToBottom {{
      0% {{ scroll-margin-top: 100vh; }}
      100% {{ scroll-margin-top: 0; }}
    }}
    
    .chat-scrollable-container {{
      animation: scrollToBottom 0.1s ease-out;
      scroll-margin-top: 0 !important;
    }}
    </style>
    <script>
    // Dynamically enable scrolling when content exceeds viewport height
    function checkScrollNeeded() {{
      const container = document.querySelector('.chat-scrollable-container');
      const column = document.querySelector('[data-testid="column"]:last-child');
      if (container && column) {{
        const containerHeight = container.scrollHeight;
        const viewportHeight = window.innerHeight;
        const hasMsgs = {'true' if has_msgs else 'false'};
        
        // Enable scrolling if content exceeds viewport height AND we have messages
        if (hasMsgs && containerHeight > viewportHeight - 100) {{
          container.classList.add('has-scroll');
          column.classList.add('has-scroll');
        }} else {{
          container.classList.remove('has-scroll');
          column.classList.remove('has-scroll');
        }}
      }}
    }}
    
    // Check initially and after each render
    if (document.readyState === 'loading') {{
      document.addEventListener('DOMContentLoaded', checkScrollNeeded);
    }} else {{
      checkScrollNeeded();
    }}
    
    // Monitor for content changes
    const observer = new MutationObserver(checkScrollNeeded);
    observer.observe(document.body, {{ childList: true, subtree: true }});
    
    // Also check on window resize
    window.addEventListener('resize', checkScrollNeeded);

    // Auto-scroll to bottom functionality
    function scrollToBottom() {{
      // Try multiple selectors to find the chat container
      const chatContainer = document.querySelector('.chat-scrollable-container') ||
                           document.querySelector('[data-testid="stMainBlockContainer"]') ||
                           document.querySelector('.main .block-container');
      
      if (chatContainer) {{
        chatContainer.scrollTop = chatContainer.scrollHeight;
        chatContainer.scrollTo({{ top: chatContainer.scrollHeight, behavior: 'smooth' }});
      }}
      
      // Also scroll main window
      window.scrollTo({{ top: document.body.scrollHeight, behavior: 'smooth' }});
    }}

    // Scroll on each message update
    const scrollObserver = new MutationObserver(function(mutations) {{
      scrollToBottom();
    }});

    // Start observing when chat loads
    function initAutoScroll() {{
      const chatContainer = document.querySelector('.chat-scrollable-container') ||
                           document.querySelector('[data-testid="stMainBlockContainer"]') ||
                           document.querySelector('.main .block-container');
      
      if (chatContainer) {{
        scrollObserver.observe(chatContainer, {{ childList: true, subtree: true, attributes: true }});
        scrollToBottom();
      }}
    }}

    // Initialize on load
    if (document.readyState === 'loading') {{
      document.addEventListener('DOMContentLoaded', initAutoScroll);
    }} else {{
      initAutoScroll();
    }}

    // Also initialize after a short delay to catch late-rendered content
    setTimeout(initAutoScroll, 500);
    setTimeout(initAutoScroll, 1500);

    // Periodically check and scroll (fallback mechanism)
    setInterval(scrollToBottom, 2000);
    </script>
    """, unsafe_allow_html=True)

    # ── Messages ──────────────────────────────────────────────────────────────
    # Create scrollable container for chat content
    st.markdown("<div class='chat-scrollable-container'>", unsafe_allow_html=True)

    # ── EMPTY STATE ────────────────────────────────────────────────────────────
    if not chat["messages"]:
        st.markdown("""
        <div class="empty-state">
            <div class="main-title">Your AI assistant for Ghana</div>
            <div class="sub-text">Smart civic insights</div>
            <div class="insight-containers-wrapper">
                <div class="insight-container">
                    <div class="border-top"></div>
                    <div class="border-right"></div>
                    <div class="border-bottom"></div>
                    <div class="border-left"></div>
                    <div class="insight-container-title">Election Results</div>
                    <div class="insight-container-desc">Access presidential and parliamentary election data across all regions of Ghana</div>
                </div>
                <div class="insight-container">
                    <div class="border-top"></div>
                    <div class="border-right"></div>
                    <div class="border-bottom"></div>
                    <div class="border-left"></div>
                    <div class="insight-container-title">Budget Analysis</div>
                    <div class="insight-container-desc">Explore national budget statements, fiscal policies, and economic indicators</div>
                </div>
                <div class="insight-container">
                    <div class="border-top"></div>
                    <div class="border-right"></div>
                    <div class="border-bottom"></div>
                    <div class="border-left"></div>
                    <div class="insight-container-title">Smart Comparison</div>
                    <div class="insight-container-desc">Compare election outcomes with budget allocations for deeper insights</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    for i, msg in enumerate(chat["messages"]):
        if msg["role"] == "user":
            intent = msg.get("intent", "")
            badge_cls = {"ELECTION": "badge-election", "BUDGET": "badge-budget",
                         "COMPARE": "badge-compare"}.get(intent, "badge-election")
            badge = f'<span class="intent-badge {badge_cls}">{intent}</span>' if intent else ""
            st.markdown(
                f'<div class="msg-user">'
                f'<div class="bubble bubble-user">{badge} {msg["content"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            intent = chat["messages"][i-1].get("intent","") if i > 0 else ""
            card_cls = {"ELECTION":"resp-card-election","BUDGET":"resp-card-budget",
                        "COMPARE":"resp-card-compare"}.get(intent, "resp-card-election")

            st.markdown(
                f'<div class="msg-bot"><div class="bubble bubble-bot">'
                f'<div class="bot-name">AcityPal</div>'
                f'{msg["content"]}</div></div>',
                unsafe_allow_html=True,
            )

            # Source citations
            if msg.get("retrieved_docs"):
                render_source_chips(msg["retrieved_docs"])



            # ── PREMIUM RAG DETAILS BUBBLE (Under every answer) ──
            st.markdown('<div class="rag-bubble-wrapper">', unsafe_allow_html=True)
            with st.expander("RAG Details", expanded=False):
                if msg.get("stage_logs"):
                    st.markdown('<div class="fp-heading">Stage Logs</div>', unsafe_allow_html=True)
                    for sl in msg["stage_logs"]:
                        st.markdown(
                            f'<div class="stage-row">'
                            f'<span class="stage-pill">{sl["stage"]}</span>'
                            f'<span class="stage-info">{" | ".join(f"{k}={v}" for k,v in sl.items() if k!="stage")}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown("<br>", unsafe_allow_html=True)
                if msg.get("retrieved_docs"):
                    st.markdown('<div class="fp-heading">Retrieved Chunks</div>', unsafe_allow_html=True)
                    for j, d in enumerate(msg["retrieved_docs"], 1):
                        src = d["metadata"].get("source","?")
                        s = d.get("score",0)
                        vs = d.get("vector_score", s)
                        ks = d.get("keyword_score",0)
                        db = d.get("domain_boost",0)
                        with st.expander(f"Chunk {j} · score={s:.4f} · {src}", expanded=False):
                            st.markdown(f"""
                                <div class="chunk-details-inner">
                                    <div class="rag-bubble-container-inner">
                                        <div class="chunk-metric-card">
                                            <div class="chunk-metric-label">Final Score</div>
                                            <div class="chunk-metric-value">{s:.4f}</div>
                                        </div>
                                        <div class="chunk-metric-card">
                                            <div class="chunk-metric-label">Vector</div>
                                            <div class="chunk-metric-value">{vs:.4f}</div>
                                        </div>
                                        <div class="chunk-metric-card">
                                            <div class="chunk-metric-label">Keyword</div>
                                            <div class="chunk-metric-value">{ks:.4f}</div>
                                        </div>
                                        <div class="chunk-metric-card">
                                            <div class="chunk-metric-label">Domain+FB</div>
                                            <div class="chunk-metric-value">{db:+.4f}</div>
                                        </div>
                                    </div>
                                    <div class="chunk-text-container">{html.escape(d["text"])}</div>
                                </div>
                            """, unsafe_allow_html=True)
                if msg.get("final_prompt"):
                    prompt_html = html.escape(msg["final_prompt"])
                    st.markdown(f"""
                        <div class="fp-section">
                            <div class="fp-heading">Final System Prompt</div>
                            <details class="fp-details">
                                <summary class="fp-summary">
                                    Click to view prompt
                                    <span class="fp-chevron">▼</span>
                                </summary>
                                <div class="final-prompt-container">{prompt_html}</div>
                            </details>
                        </div>
                    """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Feedback — aligned to RAG container edge
            st.markdown('<div class="feedback-row-wrapper">', unsafe_allow_html=True)
            fb1, fb2, fb_sp = st.columns([0.04, 0.04, 0.92], gap="small")
            with fb1:
                if st.button("👍", key=f"up_{i}", help="Good answer"):
                    doc_ids = [d.get("doc_id","") for d in msg.get("retrieved_docs",[])]
                    pipeline.feedback.record_vote(
                        query=chat["messages"][i-1]["content"] if i>0 else "",
                        doc_ids=doc_ids, vote=1, response=msg["content"]
                    )
                    st.toast("Thanks! 👍")
            with fb2:
                if st.button("👎", key=f"dn_{i}", help="Bad answer"):
                    doc_ids = [d.get("doc_id","") for d in msg.get("retrieved_docs",[])]
                    pipeline.feedback.record_vote(
                        query=chat["messages"][i-1]["content"] if i>0 else "",
                        doc_ids=doc_ids, vote=-1, response=msg["content"]
                    )
                    st.toast("Noted! We'll improve. 👎")
            st.markdown('</div>', unsafe_allow_html=True)

    # Show typing indicator when assistant is generating response
    if st.session_state.get("is_typing", False):
        st.markdown(
            '<div class="msg-bot">'
            '<div class="typing-indicator">'
            '<div class="dot"></div>'
            '<div class="dot"></div>'
            '<div class="dot"></div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)  # close scrollable container

    # Add scroll anchor at the bottom
    st.markdown('<div id="chat-bottom-anchor" style="height: 1px;"></div>', unsafe_allow_html=True)
    
    # Try to scroll to anchor using JavaScript (with fallback)
    st.markdown("""
    <script>
    // Simple scroll to anchor
    setTimeout(function() {
        var anchor = document.getElementById('chat-bottom-anchor');
        if (anchor) {
            anchor.scrollIntoView({ behavior: 'smooth', block: 'end' });
        }
    }, 100);
    </script>
    """, unsafe_allow_html=True)

    # ── Input (Streamlit chat_input = centered + fixed bottom) ───────────────
    sidebar_class = "sidebar-open" if st.session_state.show_sidebar else ""

    if st.session_state.get("edit_text_val"):
        st.info(f"🔄 **Editing Mode**: {st.session_state.edit_text_val}")
        if st.button("Cancel Edit", key="cancel_edit"):
            st.session_state.edit_text_val = None
            st.rerun()

    user_input = st.chat_input(
        "Ask me about Ghana Elections & 2025 Budget…" if not st.session_state.get("edit_text_val") else f"Editing: {st.session_state.edit_text_val[:30]}...",
        key="main_chat_input",
    )

    
    if user_input and user_input.strip():
        # Clear edit state on new submit
        st.session_state.edit_text_val = None
        utext = user_input.strip()
        intent = classify_query_intent(utext)

        # Apply region filter to query
        region = st.session_state.get("region_filter","All Regions")
        query_to_run = utext
        if region != "All Regions" and intent in ("ELECTION","COMPARE"):
            query_to_run = f"{utext} region:{region}"

        chat["messages"].append({"role":"user","content":utext,"intent":intent})

        # Store query info for answer generation and show typing indicator
        st.session_state.pending_query = query_to_run
        st.session_state.pending_intent = intent
        st.session_state.is_typing = True
        st.rerun()

    # Generate answer if typing indicator is active
    if st.session_state.get("is_typing", False) and st.session_state.get("pending_query"):
        time.sleep(3)
        with st.spinner("Retrieving context & generating answer…"):
            result = pipeline.answer(
                query=st.session_state.pending_query,
                top_k=st.session_state.top_k,
                use_hybrid=st.session_state.use_hybrid,
                prompt_variant=st.session_state.get("prompt_variant_sel","strict"),
            )
        chat["messages"].append({
            "role": "assistant",
            "content": result["response"],
            "retrieved_docs": result.get("retrieved_docs",[]),
            "final_prompt": result.get("final_prompt",""),
            "stage_logs": result.get("stage_logs",[]),
            "memory_ctx": result.get("memory_ctx",""),
        })
        if chat["title"].startswith("New Chat") and st.session_state.pending_query:
            chat["title"] = st.session_state.pending_query[:45]
        st.session_state.is_typing = False
        st.session_state.pending_query = None
        st.session_state.pending_intent = None
        st.rerun()

st.html("""
<script>
const preventScroll = (e) => {
  e.preventDefault();
};



// Add red border to sidebar column
document.addEventListener('DOMContentLoaded', function() {
  const columns = document.querySelectorAll('[data-testid="column"]');
  if (columns.length >= 2) {
    columns[0].style.borderRight = '3px solid #CE1126';
  }

  // Fix only the header elements (logo, burger, +) to be static
  function fixHeaderOnly() {
    const headerMarker = document.querySelector('.header-push-marker');
    if (headerMarker) {
      const horizontalBlock = headerMarker.closest('[data-testid="stHorizontalBlock"]');
      if (horizontalBlock) {
        horizontalBlock.style.position = 'fixed';
        horizontalBlock.style.top = '12px';
        horizontalBlock.style.left = '20px';
        horizontalBlock.style.right = '20px';
        horizontalBlock.style.zIndex = '99998';
        horizontalBlock.style.background = 'var(--bg)';
        horizontalBlock.style.backdropFilter = 'blur(12px)';
        horizontalBlock.style.webkitBackdropFilter = 'blur(12px)';
        horizontalBlock.style.border = '1px solid var(--line)';
        horizontalBlock.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.05)';
        horizontalBlock.style.padding = '12px 20px';
        horizontalBlock.style.borderRadius = '16px';
        horizontalBlock.style.marginTop = '0';
      }
    }
  }

  // Initial fix
  setTimeout(fixHeaderOnly, 100);

  // Use MutationObserver to keep header fixed
  const observer = new MutationObserver(function(mutations) {
    fixHeaderOnly();
  });

  observer.observe(document.body, { childList: true, subtree: true });
});
</script>
""")


# ── Evaluation, Analysis, Architecture tabs (backend preserved for exam) ──────

def render_evaluation_tab(pipeline):
    st.markdown("## 🔬 Critical Evaluation & Adversarial Testing")
    st.caption("Parts B · C · E — Failure cases, prompt experiments, RAG vs pure LLM")

    st.markdown("### Part B — Retrieval Failure Case")
    if st.button("▶ Run Failure Case Demo", key="run_fc"):
        with st.spinner("Demonstrating failure & fix…"):
            st.session_state.failure_data = pipeline.failure_case_demo()
    if st.session_state.failure_data:
        fd = st.session_state.failure_data
        ood = fd["out_of_domain"]
        st.code(f"Query: {ood['query']}", language="text")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="stat-card"><strong>❌ Before Fix</strong><br>', unsafe_allow_html=True)
            st.markdown(f"Max score: `{ood['before_fix']['max_score']}`")
            st.markdown(f"Relevant? {'✅' if ood['before_fix']['is_relevant'] else '❌'}")
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="stat-card"><strong>✅ After Fix</strong><br>', unsafe_allow_html=True)
            st.markdown(f"_{ood['after_fix']}_")
            st.markdown('</div>', unsafe_allow_html=True)
        st.info(ood["fix_explanation"])

    st.divider()
    st.markdown("### Part C — Prompt Comparison")
    cmp_q = st.text_input("Query:", value="What does the 2025 budget say about revenue?", key="cmpq")
    if st.button("▶ Compare 3 Prompts", key="run_pc"):
        with st.spinner("Running…"):
            st.session_state.prompt_cmp_data = pipeline.compare_prompts(cmp_q)
    if st.session_state.prompt_cmp_data:
        for v, d in st.session_state.prompt_cmp_data["variants"].items():
            st.markdown(f"**{v}** — {d['description']}")
            st.markdown(f"> {d['response'][:400]}")
            st.markdown(f"Source tag: {'✅' if d['has_source_tag'] else '❌'}")

    st.divider()
    st.markdown("### Part E — Adversarial Testing")
    if st.button("▶ Run Adversarial Queries", key="run_adv"):
        with st.spinner("Running 2 adversarial queries…"):
            st.session_state.eval_results = pipeline.run_adversarial()
    if st.session_state.eval_results:
        for r in st.session_state.eval_results:
            st.markdown(f"**{r['description']}**")
            st.code(r["query"])
            m1,m2,m3 = st.columns(3)
            m1.metric("Consistent", "✅" if r["consistency"] else "❌")
            m2.metric("Hallucination", f"{r['hallucination_rate']:.0%}")
            m3.metric("Abstained", "✅" if r["run1_abstained"] else "❌")

    st.divider()
    st.markdown("### Part E — RAG vs Pure LLM")
    if st.button("▶ Compare RAG vs Pure LLM", key="run_rl"):
        with st.spinner("Running benchmark…"):
            st.session_state.rag_vs_llm_data = pipeline.rag_vs_pure_llm()
    if st.session_state.rag_vs_llm_data:
        for item in st.session_state.rag_vs_llm_data:
            st.markdown(f"**{item['query']}**")
            c1,c2 = st.columns(2)
            with c1:
                st.markdown("**🔍 RAG**")
                st.metric("Accuracy", f"{item['rag_accuracy_proxy']:.0%}")
                st.metric("Hallucination", "🔴" if item["rag_hallucination_flag"] else "🟢")
            with c2:
                st.markdown("**🤖 Pure LLM**")
                st.metric("Accuracy", f"{item['pure_llm_accuracy_proxy']:.0%}")
                st.metric("Hallucination", "🔴" if item["pure_llm_hallucination_flag"] else "🟢")


def render_analysis_tab():
    st.markdown("## 📊 Data Engineering & Retrieval Analysis")
    from src.config import CHUNK_COMPARE, CHUNKS_A, CHUNKS_B, PIPELINE_LOG_PATH
    st.markdown("### Part A — Chunking Strategies")
    if CHUNK_COMPARE.exists():
        with open(CHUNK_COMPARE) as f:
            cmp = json.load(f)
        sa, sb = cmp.get("strategy_a",{}), cmp.get("strategy_b",{})
        cols = st.columns(5)
        for col,(lbl,key,pct) in zip(cols,[
            ("Chunks","num_chunks",False),("Avg Words","avg_chunk_words",False),
            ("KW Coverage","keyword_coverage",True),("Precision","precision_proxy",True),
            ("Density","density_score",True)]):
            va = sa.get(key,0)
            vb = sb.get(key,0)
            col.metric(f"A · {lbl}", f"{va:.0%}" if pct else va,
                       f"B: {vb:.0%}" if pct else f"B: {vb}")
        import pandas as pd
        chart = pd.DataFrame({
            "A":[sa.get("keyword_coverage",0),sa.get("precision_proxy",0),sa.get("density_score",0)],
            "B":[sb.get("keyword_coverage",0),sb.get("precision_proxy",0),sb.get("density_score",0)],
        }, index=["KW Coverage","Precision","Density"])
        st.bar_chart(chart, color=["#CE1126","#006B3F"])
        st.info(cmp.get("recommendation",""))
    else:
        st.warning("Run `python -m src.data_prep` first.")

    st.divider()
    st.markdown("### Pipeline Logs")
    if PIPELINE_LOG_PATH.exists():
        lines = [l for l in PIPELINE_LOG_PATH.read_text().split("\n") if l.strip()]
        st.metric("Logged queries", len(lines))
        if lines:
            with st.expander("Last 3 entries"):
                for l in lines[-3:]:
                    try:
                        e = json.loads(l)
                        st.json({k:v for k,v in e.items() if k!="final_prompt"})
                    except Exception:
                        st.text(l[:200])
    else:
        st.info("No logs yet — send a query in Chat first.")


def render_architecture_tab():
    from src.architecture import ARCHITECTURE_MERMAID,ARCHITECTURE_DESCRIPTION,DOMAIN_JUSTIFICATION,DATA_FLOW_STEPS
    st.markdown("## 🏗️ Architecture & System Design")
    try:
        from streamlit_mermaid import st_mermaid
        st_mermaid(ARCHITECTURE_MERMAID, height=500)
    except ImportError:
        st.code(ARCHITECTURE_MERMAID, language="mermaid")
    st.divider()
    for step in DATA_FLOW_STEPS:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:12px;padding:7px 0;"
            f"border-bottom:1px solid var(--line);'>"
            f"<span style='background:linear-gradient(90deg,#6f42ff,#00a2ff);color:white;"
            f"border-radius:50%;width:28px;height:28px;display:flex;align-items:center;"
            f"justify-content:center;font-weight:700;font-size:0.8rem;flex-shrink:0;'>{step['step']}</span>"
            f"<span style='font-weight:600;min-width:160px;'>{step['name']}</span>"
            f"<code style='background:rgba(111,66,255,0.08);padding:2px 8px;border-radius:6px;"
            f"font-size:0.8rem;color:var(--accent-1);'>{step['component']}</code>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.divider()
    st.markdown(ARCHITECTURE_DESCRIPTION)
    st.divider()
    st.markdown(DOMAIN_JUSTIFICATION)


# ── Main ───────────────────────────────────────────────────────────────────────

init_state()

# Handle query parameters for header button actions
query_params = st.query_params
if "sidebar" in query_params and query_params["sidebar"] == "open":
    st.session_state.show_sidebar = True
    st.query_params.clear()
    st.rerun()
if "new_chat" in query_params and query_params["new_chat"] == "true":
    create_new_chat()
    st.query_params.clear()
    st.rerun()
if "header_action" in query_params:
    st.session_state.header_action = query_params["header_action"]
    st.query_params.clear()
    st.rerun()

# Pre-generate Logo
logo_path = find_logo_path()
logo_data_uri = build_logo_data_uri(logo_path)
logo_html = (
    f'<img src="{logo_data_uri}" style="width:30px;height:30px;object-fit:contain;" />'
    if logo_data_uri else '<span style="font-size:16px;"></span>'
)

# ── CRITICAL BUNDLE (Styles + Landing HTML together) ───────────────────────────
if not st.session_state.started:
    # LANDING MODE: Bundle everything
    st.markdown(f"""
    <style>
        /* Un-hide only for landing */
        .stApp {{
            visibility: visible !important;
            background: linear-gradient(-45deg, #006B3F, #004d2d, #CE1126, #8b0000, #FCD116, #b8960a, #006B3F, #CE1126) !important;
            background-size: 500% 500% !important;
            animation: ghanaGrad 12s ease infinite !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        .block-container {{ visibility: visible !important; padding: 0 !important; margin: 0 !important; }}

        .gh-landing {{
          position: fixed; inset: 0;
          background: linear-gradient(-45deg, #006B3F, #004d2d, #CE1126, #8b0000, #FCD116, #b8960a, #006B3F, #CE1126);
          background-size: 500% 500%;
          animation: ghanaGrad 12s ease infinite;
          display: flex; flex-direction: column;
          align-items: center; justify-content: center;
          z-index: 9999;
        }}
        .gh-glass {{
          background: rgba(255,255,255,0.08);
          backdrop-filter: blur(24px) saturate(1.4);
          -webkit-backdrop-filter: blur(24px) saturate(1.4);
          border: 1px solid rgba(255,255,255,0.22);
          border-radius: 28px;
          padding: 48px 54px 40px;
          text-align: center;
          max-width: 640px;
          width: 90vw;
          box-shadow: 0 24px 60px rgba(0,0,0,0.35);
        }}
        .gh-logo-bar {{
          position: fixed; top: 22px; left: 26px;
          display: flex; align-items: center; gap: 9px;
          z-index: 10000;
        }}
        .gh-logo-bar img {{ width: 24px; height: 24px; object-fit: contain; }}
        .gh-logo-bar span {{
          font-size: 18px; font-weight: 800; color: #fff;
          text-shadow: 0 2px 8px rgba(0,0,0,0.4);
        }}
        .gh-title {{
          font-size: clamp(48px, 8vw, 82px);
          font-weight: 900; letter-spacing: -2px;
          color: #ffffff;
          text-shadow: 0 4px 24px rgba(0,0,0,0.4);
          line-height: 1; margin-bottom: 14px;
        }}
        .gh-subtitle {{
          color: rgba(255,255,255,0.85);
          font-size: clamp(15px, 1.6vw, 19px);
          margin-bottom: 6px; font-weight: 500;
        }}
        .typing-container {{
          height: 2.2em; position: relative;
          overflow: hidden; margin: 8px 0 28px;
        }}
        .typing-phrase {{
          position: absolute; left: 0; right: 0;
          color: #FCD116;
          font-size: clamp(14px, 1.4vw, 17px);
          font-weight: 600; opacity: 0;
          animation: phraseAnim 15s infinite;
          text-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }}
        .typing-phrase:nth-child(1) {{ animation-delay: 0s; }}
        .typing-phrase:nth-child(2) {{ animation-delay: 5s; }}
        .typing-phrase:nth-child(3) {{ animation-delay: 10s; }}
        @keyframes phraseAnim {{
          0%   {{ opacity: 0; transform: translateY(8px); }}
          8%   {{ opacity: 1; transform: translateY(0); }}
          28%  {{ opacity: 1; transform: translateY(0); }}
          35%  {{ opacity: 0; transform: translateY(-8px); }}
          100% {{ opacity: 0; }}
        }}
        .st-key-start_now_btn {{
          position: fixed !important;
          left: 50% !important;
          top: calc(50% + 220px) !important;
          transform: translateX(-50%) !important;
          z-index: 10001 !important;
          width: min(260px, 44vw) !important;
        }}
        .st-key-start_now_btn .stButton > button {{
          background: linear-gradient(135deg, #FCD116, #f0a500) !important;
          color: #111 !important; border: 2px solid transparent !important;
          font-size: 1.05rem !important; font-weight: 700 !important;
          padding: 0.78rem 0 !important;
          width: 100% !important;
          border-radius: 999px !important;
          box-shadow: 0 8px 28px rgba(252,209,22,0.55) !important;
          letter-spacing: 0.02em !important;
          transition: all 0.2s ease !important;
        }}
        .st-key-start_now_btn .stButton > button:hover {{
          transform: translateY(-2px) scale(1.03) !important;
          border-color: #CE1126 !important;
          box-shadow: 0 14px 36px rgba(252,209,22,0.65) !important;
        }}
    </style>
    <div class="gh-logo-bar">
      {logo_html}
      <span>acitypal</span>
    </div>
    <div class="gh-landing">
      <div class="gh-glass">
        <div class="gh-title">ACITYPAL</div>
        <div class="gh-subtitle">Your Academic City Intelligent Assistant<br>Ghana Elections &amp; 2025 Budget</div>
        <div class="typing-container">
          <div class="typing-phrase">❝ Who won Ashanti Region in 2020? ❞</div>
          <div class="typing-phrase">❝ What is the 2025 budget deficit target? ❞</div>
          <div class="typing-phrase">❝ Compare NPP vs NDC votes in Greater Accra. ❞</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Start Now", key="start_now_btn"):
        st.session_state.started = True
        st.session_state.has_chart = False
        if st.session_state.active_chat_idx is None:
            create_new_chat()
        st.rerun()
    st.stop()
else:
    # CHAT MODE: Un-hide content and apply final styles
    chat = get_active_chat()
    has_msgs = len(chat["messages"]) > 0 if chat else False
    inject_global_styles(allow_scroll=has_msgs)
    st.markdown("<style>.stApp, .block-container { visibility: visible !important; }</style>", unsafe_allow_html=True)
    
    pipeline = get_pipeline()
    render_fixed_header(logo_html)
    render_chat_tab(pipeline, logo_html)
