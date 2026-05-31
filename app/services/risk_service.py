"""
app/services/risk_service.py  —  NLP-based legal risk analysis engine.

9-step pipeline (per clause)
────────────────────────────
1.  Normalise text
2.  Detect clause type (safe-clause filter)
3.  Skip safe/informational clauses
4.  Check for obligation language
5.  Regex detection
6.  Semantic similarity against risk anchors
7.  Negative anchor veto (informational clause filter)
8.  Evidence threshold gate
9.  Return findings
"""
import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger("jurisai.risk")

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    _nlp = None
    SPACY_AVAILABLE = False

try:
    from sentence_transformers import util as st_util
    ST_UTIL_AVAILABLE = True
except ImportError:
    st_util = None
    ST_UTIL_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  RISK CATALOGUE
# ══════════════════════════════════════════════════════════════════════════════

RISK_CATALOGUE: List[Dict] = [
    # ── HIGH ──────────────────────────────────────────────────────────────────
    {"label": "Indemnification", "level": "high",
     "explanation": "One party must cover the other's legal costs or losses — significant open-ended financial exposure.",
     "regex": re.compile(r"\bindemnif\w*\b", re.I),
     "semantic_anchors": [
         "The party shall indemnify and hold harmless the other from any claims or losses.",
         "Indemnification obligations apply to all third-party claims.",
         "Vendor shall defend and indemnify client against all liabilities arising from service delivery.",
         "The contractor agrees to indemnify the company for any breach of representations.",
         "Each party shall indemnify the other for damages caused by its own negligence.",
         "The service provider assumes full indemnification obligations for intellectual property infringement.",
         "You agree to indemnify us from any claims resulting from your use of the platform.",
         "Indemnification shall survive termination of this agreement.",
     ]},
    {"label": "Unlimited Liability", "level": "high",
     "explanation": "No cap on financial responsibility — you could be liable for any amount.",
     "regex": re.compile(r"\bunlimited\s+liabilit\w*\b", re.I),
     "semantic_anchors": [
         "There is no limit on the liability of either party.",
         "The party accepts unlimited financial responsibility for all damages.",
         "Neither party limits its liability under any circumstance.",
         "Liability shall not be capped or restricted in any way.",
         "The company bears full and unrestricted liability for all losses.",
         "All liability including consequential and indirect damages is accepted without limit.",
     ]},
    {"label": "Liquidated Damages", "level": "high",
     "explanation": "Pre-set penalty amounts payable if certain obligations are not met.",
     "regex": re.compile(r"\bliquidated\s+damages?\b", re.I),
     "semantic_anchors": [
         "Liquidated damages shall be payable upon breach of this agreement.",
         "A fixed sum is owed as damages if the contract is not fulfilled.",
         "The parties agree to liquidated damages of a specified amount per day of delay.",
         "Upon failure to deliver, liquidated damages become immediately due and payable.",
         "Liquidated damages represent a genuine pre-estimate of loss and not a penalty.",
         "A predetermined sum shall be paid as liquidated damages for each week of non-performance.",
     ]},
    {"label": "Non-Compete", "level": "high",
     "explanation": "Restricts your ability to work in the same industry after the contract ends.",
     "regex": re.compile(r"\bnon[\s\-]?compete\b", re.I),
     "semantic_anchors": [
         "The party agrees not to compete with the other party after termination.",
         "Employee shall not work for competitors for a specified period.",
         "During the non-compete period the contractor may not engage in similar business activities.",
         "You agree not to solicit clients or employees of the company for two years.",
         "The non-competition restriction applies within a defined geographic territory.",
         "Contractor shall not directly or indirectly engage in competing business for twelve months.",
         "Post-termination restrictions prevent the party from working in the same sector.",
     ]},
    {"label": "Automatic Renewal", "level": "high",
     "explanation": "Contract renews automatically unless cancelled — you may be locked in without realising.",
     "regex": re.compile(r"\bautomatic\s+renewal\b", re.I),
     "semantic_anchors": [
         "This agreement will automatically renew unless notice of cancellation is given.",
         "Renewal happens automatically at the end of each period.",
         "The contract renews for successive one-year terms unless terminated in writing.",
         "Without prior notice of non-renewal the agreement shall automatically extend.",
         "Automatic renewal occurs thirty days before the end of the current term.",
         "The subscription continues on a rolling basis unless cancelled before the renewal date.",
     ]},
    {"label": "Unilateral Amendment", "level": "high",
     "explanation": "The other party can change contract terms without your consent.",
     "regex": re.compile(r"\bunilateral\w*\s+(amend|change|modif)\w*\b", re.I),
     "semantic_anchors": [
         "The company reserves the right to modify these terms at any time.",
         "Terms may be changed without prior consent of the other party.",
         "We may update this agreement unilaterally by posting revised terms online.",
         "The platform provider may change pricing or terms with thirty days notice.",
         "Amendments take effect immediately upon publication without requiring acceptance.",
         "The service provider retains the right to alter service terms unilaterally.",
     ]},
    {"label": "Rights Waiver", "level": "high",
     "explanation": "A broad waiver of rights may prevent you from seeking remedies later.",
     "regex": re.compile(r"\bwaives?\s+(all\s+)?rights?\b", re.I),
     "semantic_anchors": [
         "The party waives all rights to seek legal recourse.",
         "By signing, you give up your right to bring claims.",
         "The client waives any right to dispute charges after thirty days.",
         "You irrevocably waive your right to a jury trial.",
         "Signing this agreement constitutes a waiver of all prior claims.",
         "The employee waives rights to overtime compensation under applicable law.",
     ]},
    # ── MEDIUM ────────────────────────────────────────────────────────────────
    {"label": "Termination for Convenience", "level": "medium",
     "explanation": "The other party can end the contract at any time without reason.",
     "regex": re.compile(r"\btermination\s+(for\s+convenience|without\s+cause)\b", re.I),
     "semantic_anchors": [
         "Either party may terminate this agreement at any time without cause.",
         "The contract can be ended for convenience with written notice.",
         "The client may terminate for convenience upon thirty days written notice.",
         "Termination without cause is permitted subject to payment of outstanding fees.",
         "The company reserves the right to end this agreement at its sole discretion.",
         "Upon termination for convenience the contractor is entitled only to fees earned.",
     ]},
    {"label": "Confidentiality Obligation", "level": "medium",
     "explanation": "You are required to keep certain information secret — review the scope carefully.",
     "regex": re.compile(r"\bconfidentialit\w*\b", re.I),
     "semantic_anchors": [
         "The receiving party must keep all shared information confidential.",
         "Non-disclosure obligations apply to all proprietary data.",
         "All confidential information must be protected with reasonable security measures.",
         "The party agrees not to disclose trade secrets or business information to third parties.",
         "Confidentiality obligations survive the termination of this agreement for five years.",
         "The employee shall not reveal any confidential business information during or after employment.",
     ]},
    {"label": "IP Assignment", "level": "medium",
     "explanation": "Ownership of IP you create may transfer to the other party.",
     "regex": re.compile(r"\bintellectual\s+property\s+(assignment|transfer|ownership)\b", re.I),
     "semantic_anchors": [
         "All intellectual property created under this agreement belongs to the company.",
         "The contractor assigns all rights in the work to the client.",
         "Work product and inventions developed during employment are owned by the employer.",
         "The consultant hereby assigns all IP rights to the commissioning party.",
         "Any software or materials created shall be the sole property of the client.",
         "Intellectual property developed in connection with services vests in the company upon creation.",
     ]},
    {"label": "Arbitration Clause", "level": "medium",
     "explanation": "Disputes go to arbitration rather than court — understand the implications.",
     "regex": re.compile(r"\barbitration\b", re.I),
     "semantic_anchors": [
         "Any disputes shall be resolved through binding arbitration.",
         "Arbitration is the exclusive remedy for resolving disagreements.",
         "All claims must be submitted to arbitration under the rules of a named institution.",
         "The parties agree to resolve disputes through final and binding arbitration.",
         "Arbitration proceedings shall be conducted in a specified jurisdiction.",
         "By agreeing to arbitration you waive your right to a court trial.",
     ]},
    {"label": "Penalty Clause", "level": "medium",
     "explanation": "Financial penalties may apply if obligations are not met.",
     "regex": re.compile(r"\bpenalt\w*\b", re.I),
     "semantic_anchors": [
         "A financial penalty is imposed for failure to meet deadlines.",
         "The defaulting party shall pay a penalty fee.",
         "Late delivery results in a penalty of one percent of contract value per day.",
         "Breach of confidentiality triggers an immediate contractual penalty.",
         "Penalties shall be deducted from the outstanding invoice amount.",
         "A penalty charge applies for early termination of the subscription.",
     ]},
    {"label": "Exclusivity", "level": "medium",
     "explanation": "You may be restricted from working with others in the same domain.",
     "regex": re.compile(r"\bexclusive\w*\b", re.I),
     "semantic_anchors": [
         "The supplier agrees to deal exclusively with this buyer.",
         "The agreement grants exclusive rights to one party.",
         "During the exclusivity period the vendor may not supply competing businesses.",
         "You are granted exclusive distribution rights in the specified territory.",
         "The contractor shall not provide similar services to any competitor during the term.",
         "The exclusive license prohibits the licensor from granting rights to third parties.",
     ]},
    {"label": "Force Majeure", "level": "medium",
     "explanation": "Obligations may be suspended for events outside both parties' control.",
     "regex": re.compile(r"\bforce\s+majeure\b", re.I),
     "semantic_anchors": [
         "Neither party is liable for failure due to events beyond their control.",
         "Force majeure events excuse non-performance under the contract.",
         "Performance is suspended during a force majeure event without liability.",
         "Acts of God, war, or natural disasters constitute force majeure under this agreement.",
         "The force majeure clause releases obligations during extraordinary circumstances.",
         "A party invoking force majeure must notify the other party within a specified period.",
     ]},
    # ── LOW ───────────────────────────────────────────────────────────────────
    {"label": "Governing Law", "level": "low",
     "explanation": "Specifies which country/state's law applies — standard but worth noting.",
     "regex": re.compile(r"\bgoverning\s+law\b", re.I),
     "semantic_anchors": [
         "This agreement is governed by the laws of a specific jurisdiction.",
         "The parties submit to the exclusive jurisdiction of courts in a named country.",
         "Governing law and jurisdiction are set out in this clause.",
     ]},
    {"label": "Warranty", "level": "low",
     "explanation": "Promises about quality/fitness of goods or services.",
     "regex": re.compile(r"\bwarrant\w*\b", re.I),
     "semantic_anchors": [
         "The seller warrants that the product is fit for purpose.",
         "The service provider warrants that all work will be performed with reasonable skill.",
         "No warranty is given as to accuracy or completeness of information provided.",
     ]},
    {"label": "Payment Terms", "level": "low",
     "explanation": "Outlines when and how payments are to be made.",
     "regex": re.compile(r"\bpayment\s+terms?\b", re.I),
     "semantic_anchors": [
         "Payment is due within thirty days of invoice.",
         "Late payments attract interest charges.",
         "Invoices must be settled within the agreed payment period.",
     ]},
    {"label": "Dispute Resolution", "level": "low",
     "explanation": "Defines how disagreements will be handled.",
     "regex": re.compile(r"\bdispute\s+resolution\b", re.I),
     "semantic_anchors": [
         "Disputes shall first be resolved through good faith negotiation.",
         "The parties agree to attempt mediation before commencing legal proceedings.",
         "Dispute resolution procedures are set out in Schedule A.",
     ]},
]

# ── Thresholds ────────────────────────────────────────────────────────────────
SEMANTIC_THRESHOLD  = 0.62
EVIDENCE_THRESHOLD  = 4
EVIDENCE_REGEX      = 2
EVIDENCE_SEMANTIC   = 3
EVIDENCE_OBLIGATION = 2
MIN_SENTENCE_WORDS  = 6

# ── Safe clause patterns ──────────────────────────────────────────────────────
_SAFE_CLAUSE_TYPES: Dict[str, re.Pattern] = {
    "Definitions":      re.compile(r"\b(definition|defined\s+term|means\s+(?:the|a)|herein\s+defined)\b", re.I),
    "Headings":         re.compile(r"\b(heading|caption|title)s?\s+(are|is|shall\s+be)?\s*(for\s+convenience|not\s+affect|do\s+not\s+affect)", re.I),
    "Interpretation":   re.compile(r"\b(interpretation|construction)\s+(of\s+this\s+agreement|clause|section)", re.I),
    "Severability":     re.compile(r"\b(severab|if\s+any\s+(provision|clause)\s+is\s+(held|found|deemed)\s+(invalid|unenforceable))", re.I),
    "Counterparts":     re.compile(r"\bcounterpart\w*\b", re.I),
    "Entire Agreement": re.compile(r"\b(entire\s+agreement|whole\s+agreement|supersedes?\s+all\s+prior)", re.I),
    "Notice":           re.compile(r"\b(notice\s+shall\s+be\s+(given|sent|delivered)|written\s+notice\s+to)", re.I),
}

# ── Negative semantic anchors ─────────────────────────────────────────────────
_NEGATIVE_ANCHORS: List[str] = [
    "This section defines terms used in the agreement.",
    "Headings are included for convenience only and do not affect interpretation.",
    "This clause clarifies document structure and interpretation.",
    "If any provision is found invalid it shall be severed without affecting the rest.",
    "This agreement may be executed in counterparts each of which is an original.",
    "This document constitutes the entire agreement between the parties.",
    "Words in the singular include the plural and vice versa.",
    "Notice shall be in writing and delivered to the address specified.",
    "Defined terms used in this agreement have the meanings given in Schedule 1.",
    "The recitals form part of this agreement for interpretation purposes only.",
]

# ── Obligation keywords ───────────────────────────────────────────────────────
_OBLIGATION_KEYWORDS = {
    "shall", "must", "liable", "liability", "penalty", "penalt",
    "indemnif", "terminat", "compensat", "damages", "assign",
    "restrict", "prohibit", "warrant", "obligat", "covenant",
    "breach", "forfeit", "default", "enforce", "comply",
}


class RiskService:
    """
    Full 9-step NLP risk analysis pipeline.
    Precomputes anchor embeddings at startup when an EmbeddingService is provided.
    """

    def __init__(self, embedding_service=None):
        self._embedder = embedding_service
        self._anchor_embeddings: Dict[str, np.ndarray] = {}
        self._negative_embeddings: Optional[np.ndarray] = None
        self._precompute_anchors()

    # ── Precomputation ────────────────────────────────────────────────────────

    def _precompute_anchors(self):
        local_model = (
            self._embedder.local_model
            if self._embedder and hasattr(self._embedder, "local_model")
            else None
        )
        if local_model is None:
            return
        for entry in RISK_CATALOGUE:
            anchors = entry.get("semantic_anchors", [])
            if anchors:
                embs = local_model.encode(
                    anchors, convert_to_numpy=True, show_progress_bar=False
                )
                self._anchor_embeddings[entry["label"]] = embs.astype("float32")
        neg_embs = local_model.encode(
            _NEGATIVE_ANCHORS, convert_to_numpy=True, show_progress_bar=False
        )
        self._negative_embeddings = neg_embs.astype("float32")
        log.info("Pre-computed anchor embeddings for %d risk categories.",
                 len(self._anchor_embeddings))

    # ── Pipeline helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        text = re.sub(r"[\u2018\u2019]", "'", text)
        text = re.sub(r"[\u201c\u201d]", '"', text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _detect_clause_type(chunk: str) -> Optional[str]:
        for clause_type, pattern in _SAFE_CLAUSE_TYPES.items():
            if pattern.search(chunk):
                return clause_type
        return None

    @staticmethod
    def _has_obligation_language(text: str) -> bool:
        lower = text.lower()
        return any(kw in lower for kw in _OBLIGATION_KEYWORDS)

    def _obligation_sentences(self, text: str) -> List[str]:
        if SPACY_AVAILABLE and _nlp:
            doc = _nlp(text[:50000])
            sents = [s.text.strip() for s in doc.sents]
        else:
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)]
        return [
            s for s in sents
            if len(s.split()) >= MIN_SENTENCE_WORDS and self._has_obligation_language(s)
        ]

    def _is_semantically_safe(self, sentences: List[str]) -> bool:
        local_model = (
            self._embedder.local_model
            if self._embedder and hasattr(self._embedder, "local_model")
            else None
        )
        if local_model is None or not ST_UTIL_AVAILABLE:
            return False
        if self._negative_embeddings is None or not sentences:
            return False
        sent_embs = local_model.encode(
            sentences, convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")
        neg_sim = float(st_util.cos_sim(sent_embs, self._negative_embeddings).numpy().max())
        best_risk_sim = 0.0
        for embs in self._anchor_embeddings.values():
            sim = float(st_util.cos_sim(sent_embs, embs).numpy().max())
            if sim > best_risk_sim:
                best_risk_sim = sim
        return neg_sim > best_risk_sim

    def _semantic_match(self, sentences: List[str], label: str) -> Optional[str]:
        local_model = (
            self._embedder.local_model
            if self._embedder and hasattr(self._embedder, "local_model")
            else None
        )
        if local_model is None or not ST_UTIL_AVAILABLE:
            return None
        if not sentences or label not in self._anchor_embeddings:
            return None
        sentences = [s for s in sentences if len(s.split()) >= MIN_SENTENCE_WORDS]
        if not sentences:
            return None
        anchor_embs = self._anchor_embeddings[label]
        sent_embs = local_model.encode(
            sentences, convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")
        sim = st_util.cos_sim(sent_embs, anchor_embs).numpy()
        best_i = int(np.argmax(sim.max(axis=1)))
        best_s = float(sim[best_i].max())
        if best_s >= SEMANTIC_THRESHOLD:
            return re.sub(r"\s+", " ", sentences[best_i]).strip()
        return None

    # ── Clause-level analyser ─────────────────────────────────────────────────

    def _analyse_clause(self, chunk: str) -> List[Dict[str, Any]]:
        chunk = self._normalize(chunk)
        clause_type = self._detect_clause_type(chunk)
        if clause_type:
            return []
        has_obligation = self._has_obligation_language(chunk)
        ob_sents = self._obligation_sentences(chunk)
        if ob_sents and self._is_semantically_safe(ob_sents):
            return []

        findings: List[Dict[str, Any]] = []
        for entry in RISK_CATALOGUE:
            label       = entry["label"]
            level       = entry["level"]
            explanation = entry["explanation"]
            regex       = entry.get("regex")
            evidence    = 0
            detected_by = []
            excerpt     = ""

            if regex:
                m = regex.search(chunk)
                if m:
                    evidence += EVIDENCE_REGEX
                    detected_by.append("regex")
                    s   = max(0, m.start() - 100)
                    e   = min(len(chunk), m.end() + 120)
                    raw = chunk[s:e].strip()
                    excerpt = "..." + re.sub(r"\s+", " ", raw)[:217] + "..."

            if has_obligation:
                evidence += EVIDENCE_OBLIGATION

            if ob_sents:
                sem = self._semantic_match(ob_sents, label)
                if sem:
                    evidence += EVIDENCE_SEMANTIC
                    detected_by = (
                        ["regex+semantic"] if "regex" in detected_by else ["semantic"]
                    )
                    excerpt = "..." + sem[:217] + "..."

            if evidence < EVIDENCE_THRESHOLD:
                continue

            effective_level = level
            if level == "high" and not has_obligation:
                effective_level = "medium"
                explanation += " (downgraded: no obligation language detected)"

            findings.append({
                "label":       label,
                "level":       effective_level,
                "explanation": explanation,
                "excerpt":     excerpt,
                "detected_by": detected_by[0] if detected_by else "regex",
                "evidence":    evidence,
            })
        return findings

    # ── Public entry point ────────────────────────────────────────────────────

    def analyze(self, chunks: List[str]) -> Dict[str, Any]:
        """Analyse a list of clause chunks and return a consolidated risk report."""
        WEIGHTS = {"high": 15, "medium": 7, "low": 2}
        best: Dict[str, Dict] = {}
        for chunk in chunks:
            for finding in self._analyse_clause(chunk):
                label = finding["label"]
                if label not in best or finding["evidence"] > best[label]["evidence"]:
                    best[label] = finding

        findings = sorted(
            best.values(),
            key=lambda f: {"high": 0, "medium": 1, "low": 2}.get(f["level"], 3),
        )
        for f in findings:
            f.pop("evidence", None)

        score       = min(100, sum(WEIGHTS.get(f["level"], 0) for f in findings))
        level_label = "High" if score >= 65 else ("Medium" if score >= 35 else "Low")
        hc = sum(1 for f in findings if f["level"] == "high")
        mc = sum(1 for f in findings if f["level"] == "medium")
        lc = sum(1 for f in findings if f["level"] == "low")
        sc = sum(1 for f in findings if "semantic" in f.get("detected_by", ""))

        return {
            "risk_score":    score,
            "risk_level":    level_label,
            "findings":      findings,
            "summary":       (
                f"Overall risk: {level_label} (score {score}/100). "
                f"Found {hc} high-risk, {mc} medium-risk, {lc} low-risk clause(s). "
                f"{sc} clause(s) confirmed via semantic NLP."
            ),
            "nlp_available": SPACY_AVAILABLE and bool(self._anchor_embeddings),
        }
