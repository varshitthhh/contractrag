import re

def _tokenize(text):
    pat = r'[a-z]{3,}'
    return set(re.findall(pat, text.lower()))

def compute_scorecard(answer, chunks, citations):
    try:
        scores = [c.get('score', 0.0) if isinstance(c, dict) else getattr(c, 'score', 0.0) for c in citations]
        retrieval_score = round(sum(scores)/len(scores), 3) if scores else 0.0
        ctx = set()
        for ch in chunks:
            ctx |= _tokenize(ch.get('text', '') if isinstance(ch, dict) else str(ch))
        atoks = _tokenize(answer)
        grounding_score = round(len(atoks & ctx)/len(atoks), 3) if atoks else 0.0
        faithfulness_flag = grounding_score >= 0.55 and retrieval_score >= 0.3
        return {'retrieval_score': retrieval_score, 'grounding_score': grounding_score, 'faithfulness_flag': faithfulness_flag, 'num_chunks_used': len(chunks)}
    except Exception as e:
        return {'retrieval_score': 0.0, 'grounding_score': 0.0, 'faithfulness_flag': False, 'num_chunks_used': 0}
