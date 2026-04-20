import json
import re
from typing import Dict, List
from langchain_core.documents import Document
from langchain_core.tools import tool
from db.parent_store_manager import ParentStoreManager

MAX_EXPANDED_QUERIES = 3
MIN_PER_QUERY_LIMIT = 6
RETRIEVAL_SCORE_THRESHOLD = 0.6
RRF_K = 60
FUSION_METHOD = "per_query_qdrant_hybrid+rrf+max_score+query_coverage"
EN_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "what", "when",
    "where", "which", "who", "why", "with",
}
ZH_FILLER_PHRASES = [
    "请问", "什么是", "是什么", "有哪些", "有哪几种", "如何", "怎么", "为什么",
    "请介绍", "介绍一下", "总结一下", "说明一下",
]

class ToolFactory:
    
    def __init__(self, collection):
        self.collection = collection
        self.parent_store_manager = ParentStoreManager()

    @staticmethod
    def _dedupe_strings(items: List[str]) -> List[str]:
        seen = set()
        result = []
        for item in items:
            normalized = item.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
        return result

    @staticmethod
    def _normalize_query(query: str) -> str:
        return re.sub(r"\s+", " ", query.strip())

    def _build_keyword_query(self, query: str) -> str:
        cleaned = self._normalize_query(query)
        for phrase in ZH_FILLER_PHRASES:
            cleaned = cleaned.replace(phrase, " ")
        cleaned = re.sub(r"[?？!！,，。；;：:（）()\[\]{}\"'`]+", " ", cleaned)

        tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", cleaned.lower())
        keywords = []
        for token in tokens:
            if re.fullmatch(r"[A-Za-z0-9_]+", token):
                if token in EN_STOPWORDS or len(token) <= 2:
                    continue
            elif len(token) <= 1:
                continue
            keywords.append(token)

        return " ".join(self._dedupe_strings(keywords)[:10])

    def _expand_queries(self, query: str) -> List[str]:
        normalized = self._normalize_query(query)
        candidates = [normalized]

        keyword_query = self._build_keyword_query(normalized)
        if keyword_query and keyword_query.lower() != normalized.lower():
            candidates.append(keyword_query)

        keyword_tokens = keyword_query.split()
        if len(keyword_tokens) >= 4:
            focus_query = " ".join(self._dedupe_strings(keyword_tokens[:3] + keyword_tokens[-2:]))
            if focus_query and focus_query.lower() not in {c.lower() for c in candidates}:
                candidates.append(focus_query)

        return candidates[:MAX_EXPANDED_QUERIES]

    @staticmethod
    def _get_hit_key(doc: Document) -> str:
        parent_id = str(doc.metadata.get("parent_id", "")).strip()
        if parent_id:
            return f"parent::{parent_id}"
        source = str(doc.metadata.get("source", "unknown")).strip()
        snippet = re.sub(r"\s+", " ", doc.page_content.strip())[:120]
        return f"source::{source}::{snippet}"

    def _merge_hits(self, raw_hits: List[Dict], limit: int) -> List[Dict]:
        grouped_hits: Dict[str, Dict] = {}

        for hit in raw_hits:
            doc = hit["doc"]
            key = self._get_hit_key(doc)
            current = grouped_hits.get(key)

            if current is None:
                grouped_hits[key] = {
                    "doc": doc,
                    "parent_id": str(doc.metadata.get("parent_id", "")).strip(),
                    "source_name": str(doc.metadata.get("source", "unknown")).strip(),
                    "rrf_score": 1.0 / (RRF_K + hit["rank"]),
                    "max_score": float(hit["score"]),
                    "best_score": float(hit["score"]),
                    "query_matches": [hit["query"]],
                }
                continue

            current["rrf_score"] += 1.0 / (RRF_K + hit["rank"])
            current["max_score"] = max(current["max_score"], float(hit["score"]))
            current["query_matches"] = self._dedupe_strings(current["query_matches"] + [hit["query"]])

            if float(hit["score"]) > current["best_score"]:
                current["doc"] = doc
                current["best_score"] = float(hit["score"])

        merged_hits = []
        for item in grouped_hits.values():
            query_coverage_bonus = 0.05 * len(item["query_matches"])
            final_score = item["rrf_score"] + (0.2 * item["max_score"]) + query_coverage_bonus
            merged_hits.append({
                **item,
                "final_score": final_score,
            })

        merged_hits.sort(
            key=lambda item: (item["final_score"], item["max_score"], len(item["doc"].page_content)),
            reverse=True,
        )
        return merged_hits[:limit]

    def _format_retrieval_header(self, payload: Dict) -> str:
        return (
            "[RETRIEVAL_ENHANCEMENT]\n"
            f"{json.dumps(payload, ensure_ascii=False)}\n"
            "[/RETRIEVAL_ENHANCEMENT]"
        )

    def _format_merged_results(self, original_query: str, expanded_queries: List[str], raw_hits: List[Dict], merged_hits: List[Dict]) -> str:
        header = self._format_retrieval_header({
            "original_query": original_query,
            "expanded_queries": expanded_queries,
            "query_count": len(expanded_queries),
            "raw_hit_count": len(raw_hits),
            "deduped_hit_count": len(merged_hits),
            "fusion_method": FUSION_METHOD,
            "top_source_names": self._dedupe_strings([hit["source_name"] for hit in merged_hits])[:6],
            "top_parent_ids": self._dedupe_strings([hit["parent_id"] for hit in merged_hits if hit["parent_id"]])[:6],
        })

        if not merged_hits:
            return f"{header}\n\nNO_RELEVANT_CHUNKS"

        formatted_hits = "\n\n".join([
            f"Parent ID: {hit['parent_id']}\n"
            f"File Name: {hit['source_name']}\n"
            f"Content: {hit['doc'].page_content.strip()}"
            for hit in merged_hits
        ])
        return f"{header}\n\n{formatted_hits}"
    
    def _search_child_chunks(self, query: str, limit: int) -> str:
        """Search for the top K most relevant child chunks.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
        """
        try:
            expanded_queries = self._expand_queries(query)
            per_query_limit = max(limit * 2, MIN_PER_QUERY_LIMIT)
            raw_hits = []

            for expanded_query in expanded_queries:
                results = self.collection.similarity_search_with_score(
                    expanded_query,
                    k=per_query_limit,
                    score_threshold=RETRIEVAL_SCORE_THRESHOLD,
                )
                for rank, (doc, score) in enumerate(results, start=1):
                    raw_hits.append({
                        "query": expanded_query,
                        "rank": rank,
                        "score": score,
                        "doc": doc,
                    })

            merged_hits = self._merge_hits(raw_hits, limit=limit)
            return self._format_merged_results(query, expanded_queries, raw_hits, merged_hits)
        except Exception as e:
            return f"RETRIEVAL_ERROR: {str(e)}"
    
    def _retrieve_many_parent_chunks(self, parent_ids: List[str]) -> str:
        """Retrieve full parent chunks by their IDs.
    
        Args:
            parent_ids: List of parent chunk IDs to retrieve
        """
        try:
            ids = [parent_ids] if isinstance(parent_ids, str) else list(parent_ids)
            raw_parents = self.parent_store_manager.load_content_many(ids)
            if not raw_parents:
                return "NO_PARENT_DOCUMENTS"

            return "\n\n".join([
                f"Parent ID: {doc.get('parent_id', 'n/a')}\n"
                f"File Name: {doc.get('metadata', {}).get('source', 'unknown')}\n"
                f"Content: {doc.get('content', '').strip()}"
                for doc in raw_parents
            ])            

        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"
    
    def _retrieve_parent_chunks(self, parent_id: str) -> str:
        """Retrieve full parent chunks by their IDs.
    
        Args:
            parent_id: Parent chunk ID to retrieve
        """
        try:
            parent = self.parent_store_manager.load_content(parent_id)
            if not parent:
                return "NO_PARENT_DOCUMENT"

            return (
                f"Parent ID: {parent.get('parent_id', 'n/a')}\n"
                f"File Name: {parent.get('metadata', {}).get('source', 'unknown')}\n"
                f"Content: {parent.get('content', '').strip()}"
            )          

        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"
    
    def create_tools(self) -> List:
        """Create and return the list of tools."""
        search_tool = tool("search_child_chunks")(self._search_child_chunks)
        retrieve_tool = tool("retrieve_parent_chunks")(self._retrieve_parent_chunks)
        
        return [search_tool, retrieve_tool]
