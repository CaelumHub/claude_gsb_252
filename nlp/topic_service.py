"""话题聚类模型的持久化与增量更新服务。

语料在 :class:`ShardedStore` 中按分片存放。服务为每个分片缓存文件指纹，
新增、删除或修改后只重新解析变化分片，只重新分词受影响文档；未变化分片和
文档的向量、簇归属会被复用。

模型状态使用原子 JSON 文件保存，并通过独立文件锁串行化多个 Flask 线程/进程
的更新。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from collections import Counter
from typing import Optional

from nlp.segmenter import Segmenter
from nlp.topics import (
    TfidfModel,
    average_vectors,
    cosine_sparse,
    estimate_topic_count,
    make_label,
    multidimensional_scaling,
    normalize_sparse,
    spherical_kmeans,
    tokenize_text,
    two_means,
)
from storage import ShardedStore
from storage.lock import FileLock


DEFAULT_MAX_TOPICS = 8
_MIN_SPLIT_SIZE = 5


def _atomic_write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)


def _read_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _round_vec(vec: dict[str, float], digits: int = 8) -> dict[str, float]:
    return {term: round(value, digits) for term, value in vec.items()
            if abs(value) > 10 ** (-digits - 1)}


class TopicModelService:
    """管理当前语料上的一个话题聚类模型。"""

    def __init__(self, corpus_store: ShardedStore, state_path: str,
                 segmenter: Optional[Segmenter] = None):
        self.corpus = corpus_store
        self.state_path = state_path
        self.lock_path = state_path + ".lock"
        self.segmenter = segmenter or Segmenter()
        self.state = self._empty_state()
        if os.path.exists(state_path):
            self.state = _read_json(state_path, self.state)

    @staticmethod
    def _empty_state() -> dict:
        return {
            "version": 1,
            "config": {"requested_topics": None, "max_topics": DEFAULT_MAX_TOPICS},
            "documents": {},
            "clusters": {},
            "document_frequency": {},
            "shard_fingerprints": {},
            "updated_at": None,
            "built_at": None,
        }

    # -- 对外 API ---------------------------------------------------------
    def build(self, requested_topics: Optional[int] = None,
              max_topics: int = DEFAULT_MAX_TOPICS,
              force: bool = False) -> dict:
        """构建或刷新模型。force=True 时即使语料未变也完整重算。"""
        with FileLock(self.lock_path):
            self.state = _read_json(self.state_path, self._empty_state())
            return self._build_locked(requested_topics, max_topics, force)

    def _build_locked(self, requested_topics: Optional[int],
                      max_topics: int, force: bool) -> dict:
        old_config = self.state.get("config", {})
        requested = self._normalize_requested(requested_topics)
        max_topics = max(2, min(int(max_topics or DEFAULT_MAX_TOPICS), 12))
        config_changed = (
            force or
            old_config.get("requested_topics") != requested or
            int(old_config.get("max_topics", max_topics)) != max_topics
        )

        scan = self._scan_corpus(rescan_all=force)
        if not self.state.get("clusters") or force or config_changed:
            # 分片只做文件级读取，分词仍复用缓存；模型重建时需覆盖全部活文档。
            full_scan = self._scan_corpus(rescan_all=True)
            changes = {
                "added": len(full_scan["active"]),
                "updated": 0,
                "deleted": 0,
                "moved": 0,
                "rescanned_shards": len(full_scan["fingerprints"]),
                "rebuilt": True,
            }
            self._full_build(
                full_scan["active"], requested, max_topics,
                reuse_tokens=not force,
                fingerprints=full_scan["fingerprints"],
            )
        else:
            changes = self._incremental_update(scan, requested, max_topics)
        self.state["updated_at"] = time.time()
        self._save()
        result = self.snapshot()
        result["changes"] = changes
        return result

    def sync(self) -> dict:
        """按当前配置同步语料变化；尚未建模时自动建模。"""
        with FileLock(self.lock_path):
            self.state = _read_json(self.state_path, self._empty_state())
            cfg = self.state.get("config", {})
            return self._build_locked(
                requested_topics=cfg.get("requested_topics"),
                max_topics=cfg.get("max_topics", DEFAULT_MAX_TOPICS),
                force=False,
            )

    def snapshot(self, doc_limit: int = 100, doc_offset: int = 0) -> dict:
        """返回页面所需的簇、关键词、代表文档和簇间距离。"""
        docs = self.state["documents"]
        clusters = self.state["clusters"]
        df = self.state["document_frequency"]
        n_docs = len(docs)
        ordered = self._ordered_cluster_ids()
        center_by_id = {cid: clusters[cid].get("centroid", {}) for cid in ordered}
        distances = self._distance_matrix(ordered, center_by_id)
        coords = multidimensional_scaling(distances, 2) if ordered else []

        cluster_summaries = []
        for idx, cid in enumerate(ordered):
            cluster = clusters[cid]
            members = self._members(cid)
            members.sort(key=lambda doc: (-doc.get("score", 0.0), doc["id"]))
            representatives = [self._doc_brief(doc, preview=True)
                               for doc in members[:5]]
            visible = members[doc_offset:doc_offset + doc_limit] if doc_limit > 0 else []
            cohesion = cluster.get("cohesion", 0.0)
            cluster_summaries.append({
                "id": cid,
                "label": cluster.get("label") or "未命名话题",
                "keywords": cluster.get("keywords", []),
                "size": len(members),
                "cohesion": round(cohesion, 4),
                "x": round(coords[idx][0], 4) if idx < len(coords) else 0,
                "y": round(coords[idx][1], 4) if idx < len(coords) else 0,
                "representatives": representatives,
                "documents": [self._doc_brief(doc, preview=True) for doc in visible],
                "document_total": len(members),
            })

        cfg = self.state.get("config", {})
        return {
            "topics": len(ordered),
            "requested_topics": cfg.get("requested_topics"),
            "auto": cfg.get("requested_topics") is None,
            "max_topics": cfg.get("max_topics", DEFAULT_MAX_TOPICS),
            "document_count": n_docs,
            "clusters": cluster_summaries,
            "distances": [
                {"source": ordered[i], "target": ordered[j],
                 "distance": round(distances[i][j], 4),
                 "similarity": round(1 - distances[i][j], 4)}
                for i in range(len(ordered))
                for j in range(i + 1, len(ordered))
            ],
            "updated_at": self.state.get("updated_at"),
            "built_at": self.state.get("built_at"),
        }

    def get_cluster(self, cluster_id: str, limit: int = 50,
                    offset: int = 0) -> Optional[dict]:
        clusters = self.state.get("clusters", {})
        if cluster_id not in clusters:
            return None
        members = self._members(cluster_id)
        members.sort(key=lambda doc: (-doc.get("score", 0.0), doc["id"]))
        page = members[offset:offset + limit]
        cluster = clusters[cluster_id]
        return {
            "id": cluster_id,
            "label": cluster.get("label", "未命名话题"),
            "keywords": cluster.get("keywords", []),
            "size": len(members),
            "cohesion": round(cluster.get("cohesion", 0.0), 4),
            "representatives": [self._doc_brief(doc, preview=True)
                                for doc in members[:5]],
            "documents": [self._doc_brief(doc, preview=True) for doc in page],
            "total": len(members),
            "limit": limit,
            "offset": offset,
        }

    # -- 分片扫描 ---------------------------------------------------------
    def _scan_corpus(self, rescan_all: bool = False) -> dict:
        old_fps = {str(k): v for k, v in self.state.get("shard_fingerprints", {}).items()}
        current_fps = {str(item["index"]): item
                       for item in self.corpus.shard_fingerprints()}
        changed = set(current_fps) if rescan_all else {
            idx for idx, fp in current_fps.items()
            if idx not in old_fps or old_fps[idx] != fp
        }
        # 分片被删除/压缩导致编号消失时，旧编号也视为变化。
        changed.update(idx for idx in old_fps if idx not in current_fps)

        active: dict[str, dict] = {}
        for idx in sorted(changed, key=lambda x: int(x)):
            if idx not in current_fps:
                continue
            for record in self.corpus.read_shard(int(idx)):
                if record.get("_deleted") or not record.get("text"):
                    continue
                active[record["id"]] = self._record_meta(record, int(idx))
        return {"active": active, "changed_shards": changed,
                "fingerprints": current_fps}

    @staticmethod
    def _record_meta(record: dict, shard: int) -> dict:
        text = record.get("text", "")
        return {
            "id": record["id"],
            "name": record.get("name") or record["id"],
            "text": text,
            "text_hash": _text_hash(text),
            "shard": shard,
            "created_at": record.get("created_at") or record.get("_created", 0),
        }

    # -- 完整构建 ---------------------------------------------------------
    def _full_build(self, scanned: dict[str, dict], requested: Optional[int],
                    max_topics: int, reuse_tokens: bool = False,
                    fingerprints: Optional[dict] = None) -> None:
        old_docs = self.state.get("documents", {}) if reuse_tokens else {}
        documents: dict[str, dict] = {}
        df: dict[str, int] = {}

        # 全量构建时扫描所有存活语料（分片仍按指纹逐个读取，分词可复用缓存）。
        all_records = scanned

        for doc_id, meta in all_records.items():
            old = old_docs.get(doc_id, {})
            counts = old.get("term_counts") if old.get("text_hash") == meta["text_hash"] else None
            if counts is None:
                tokens = tokenize_text(meta["text"], self.segmenter)
                counts = dict(Counter(tokens))
            doc = self._base_doc(meta, counts)
            documents[doc_id] = doc
            for term in counts:
                df[term] = df.get(term, 0) + 1

        self.state["documents"] = documents
        self.state["document_frequency"] = df
        self.state["clusters"] = {}
        self.state["config"] = {"requested_topics": requested,
                                "max_topics": max_topics}

        if documents:
            model = TfidfModel(df, len(documents))
            ids = sorted(documents)
            vectors = [model.vectorize(documents[doc_id]["term_counts"])
                       for doc_id in ids]
            for doc_id, vec in zip(ids, vectors):
                documents[doc_id]["vector"] = _round_vec(vec)
            k = self._choose_full_k(vectors, requested, max_topics)
            labels, centers = spherical_kmeans(vectors, k)
            temp_ids = [f"tmp_{i}" for i in range(k)]
            for doc_id, label, vec in zip(ids, labels, vectors):
                self._assign_to_cluster(doc_id, temp_ids[label], vec,
                                        centers[label], save_score=False)
            self._stabilize_cluster_ids(temp_ids)
            self._refresh_all_cluster_metadata()
        else:
            self.state["clusters"] = {}
        self.state["built_at"] = time.time()
        self.state["shard_fingerprints"] = fingerprints or self._current_fingerprints()

    def _choose_full_k(self, vectors, requested, max_topics: int) -> int:
        n = len(vectors)
        if requested is not None:
            return max(1, min(requested, n))
        return estimate_topic_count(
            vectors, max_clusters=max(1, min(max_topics, n)))

    # -- 增量更新 ---------------------------------------------------------
    def _incremental_update(self, scan: dict, requested: Optional[int],
                            max_topics: int) -> dict:
        docs = self.state["documents"]
        active_scanned = scan["active"]
        changed_shards = scan["changed_shards"]

        old_scanned_ids = {doc_id for doc_id, doc in docs.items()
                           if str(doc.get("shard")) in changed_shards}
        deleted_ids = old_scanned_ids - set(active_scanned)
        added_ids = [doc_id for doc_id in active_scanned if doc_id not in docs]
        updated_ids = []
        moved_ids = []
        for doc_id, meta in active_scanned.items():
            old = docs.get(doc_id)
            if not old:
                continue
            if old.get("text_hash") != meta["text_hash"]:
                updated_ids.append(doc_id)
            elif old.get("shard") != meta["shard"]:
                moved_ids.append(doc_id)

        if not (added_ids or updated_ids or deleted_ids or moved_ids):
            no_changes = {
                "added": 0, "updated": 0, "deleted": 0, "moved": 0,
                "rescanned_shards": 0, "rebuilt": False,
            }
            result = self.snapshot()
            result["changes"] = no_changes
            return result

        affected_clusters: set[str] = set()
        # 旧归属和旧词频先移除。
        for doc_id in list(deleted_ids):
            affected_clusters.add(docs[doc_id].get("cluster"))
            self._detach_document(doc_id)
            docs.pop(doc_id, None)
        for doc_id in updated_ids:
            affected_clusters.add(docs[doc_id].get("cluster"))
            self._detach_document(doc_id)

        # 更新分片字段（同内容移动无需重新分词）。
        for doc_id in moved_ids:
            docs[doc_id]["shard"] = active_scanned[doc_id]["shard"]

        updated_clusters = {doc_id: docs[doc_id].get("cluster") for doc_id in updated_ids}

        # 新/改文档分词；旧文档的 DF 已在 detach 阶段移除，这里只加入新 DF。
        changed_for_assignment = list(added_ids) + list(updated_ids)
        new_counts: dict[str, dict] = {}
        for doc_id in changed_for_assignment:
            meta = active_scanned[doc_id]
            counts = dict(Counter(tokenize_text(meta["text"], self.segmenter)))
            new_counts[doc_id] = counts
            self._apply_df_delta(counts, 1)
            if doc_id in updated_clusters:
                docs[doc_id].update(self._base_doc(meta, counts))
                docs[doc_id]["cluster"] = updated_clusters[doc_id]
            else:
                docs[doc_id] = self._base_doc(meta, counts)

        # 先记录变化前质心；随后 IDF 重算和新文档落簇会改变当前质心，拆分时仍
        # 需要用它判断哪一半延续旧话题。
        previous_centroids = {
            cid: dict(cluster.get("centroid", {}))
            for cid, cluster in self.state["clusters"].items()
        }
        # IDF 是全局统计：复用缓存词频重算旧向量，不重新分词。固定 k 时先保留
        # 删除造成的空簇，给随后迁移的修改文档一个目标簇。
        self._refresh_old_vectors(remove_empty=requested is None)
        model = TfidfModel(self.state["document_frequency"], len(docs))
        for doc_id in changed_for_assignment:
            docs[doc_id]["vector"] = _round_vec(model.vectorize(new_counts[doc_id]))

        # 新文档先落入最近旧簇，修改文档保留原簇；随后的拆分/局部修正再把新主题析出。
        for doc_id in changed_for_assignment:
            doc = docs[doc_id]
            if doc.get("cluster") is None:
                cid, score, second = self._nearest_cluster(doc["vector"])
                if cid is None:
                    cid = self._create_cluster()
                doc["cluster"] = cid
                doc["score"] = score
                doc["ambiguity"] = round(second - score, 6)
            self._add_to_cluster(doc_id, doc["cluster"], doc)
            affected_clusters.add(doc["cluster"])

        target_k = requested
        if target_k is None:
            self._remove_empty_clusters()
            self._auto_adjust_clusters(max_topics, previous_centroids)

        if changed_for_assignment:
            self._reassign_changed_documents(changed_for_assignment)
            if target_k is None:
                self._auto_adjust_clusters(max_topics, previous_centroids)
            affected_clusters.update(self.state["clusters"])

        if target_k is not None:
            self._remove_empty_clusters()
            self._adjust_to_fixed_k(target_k, previous_centroids)

        # 对受影响簇的成员做局部球面 k-means，允许边界文档在这些簇间移动。
        affected_clusters = {cid for cid in affected_clusters
                             if cid in self.state["clusters"]}
        if affected_clusters:
            self._refine_clusters(affected_clusters)
        if target_k is None:
            self._auto_merge_remote_singletons()
        self._canonicalize_cluster_ids()
        affected_clusters = set(self.state["clusters"])
        # IDF 会随增删改变，所有标签都应按新词权重刷新。
        self._refresh_all_cluster_metadata()
        self.state["config"] = {"requested_topics": requested,
                                "max_topics": max_topics}
        self.state["shard_fingerprints"] = scan["fingerprints"]

        return {
            "added": len(added_ids),
            "updated": len(updated_ids),
            "deleted": len(deleted_ids),
            "moved": len(moved_ids),
            "rescanned_shards": len(changed_shards),
            "rebuilt": False,
        }

    def _apply_df_delta(self, counts: dict[str, int], delta: int) -> None:
        df = self.state["document_frequency"]
        for term in counts:
            df[term] = max(0, df.get(term, 0) + delta)
            if df[term] == 0:
                df.pop(term, None)

    # -- 簇统计原语 -------------------------------------------------------
    def _base_doc(self, meta: dict, counts: dict[str, int]) -> dict:
        return {
            "id": meta["id"],
            "name": meta["name"],
            "text_hash": meta["text_hash"],
            "shard": meta["shard"],
            "created_at": meta.get("created_at", 0),
            "preview": meta.get("text", "")[:180],
            "term_counts": counts,
            "length": sum(counts.values()),
            "vector": {},
            "cluster": None,
            "score": 0.0,
            "ambiguity": 0.0,
        }

    def _new_cluster_blob(self) -> dict:
        return {
            "centroid": {},
            "sum_vector": {},
            "size": 0,
            "term_freq": {},
            "term_doc_freq": {},
            "keywords": [],
            "label": "未命名话题",
            "cohesion": 0.0,
            "created_at": time.time(),
        }

    def _create_cluster(self) -> str:
        cid = f"new_{uuid.uuid4().hex[:10]}"
        self.state["clusters"][cid] = self._new_cluster_blob()
        return cid

    def _add_to_cluster(self, doc_id: str, cid: str, doc: Optional[dict] = None) -> None:
        doc = doc or self.state["documents"][doc_id]
        clusters = self.state["clusters"]
        if cid not in clusters:
            clusters[cid] = self._new_cluster_blob()
        cluster = clusters[cid]
        cluster["size"] += 1
        for term, value in doc.get("vector", {}).items():
            cluster["sum_vector"][term] = cluster["sum_vector"].get(term, 0.0) + value
        for term, count in doc.get("term_counts", {}).items():
            cluster["term_freq"][term] = cluster["term_freq"].get(term, 0) + count
            cluster["term_doc_freq"][term] = cluster["term_doc_freq"].get(term, 0) + 1
        doc["cluster"] = cid
        self._normalize_cluster_center(cid)

    def _remove_from_cluster(self, doc_id: str, cid: str) -> None:
        docs = self.state["documents"]
        cluster = self.state["clusters"].get(cid)
        doc = docs.get(doc_id)
        if not cluster or not doc:
            return
        cluster["size"] = max(0, cluster["size"] - 1)
        for term, value in doc.get("vector", {}).items():
            cluster["sum_vector"][term] = cluster["sum_vector"].get(term, 0.0) - value
            if abs(cluster["sum_vector"][term]) < 1e-10:
                cluster["sum_vector"].pop(term, None)
        for term, count in doc.get("term_counts", {}).items():
            cluster["term_freq"][term] = max(0, cluster["term_freq"].get(term, 0) - count)
            if cluster["term_freq"][term] == 0:
                cluster["term_freq"].pop(term, None)
            cluster["term_doc_freq"][term] = max(
                0, cluster["term_doc_freq"].get(term, 0) - 1)
            if cluster["term_doc_freq"][term] == 0:
                cluster["term_doc_freq"].pop(term, None)
        doc["cluster"] = None
        self._normalize_cluster_center(cid)

    def _detach_document(self, doc_id: str) -> None:
        """从簇和全局索引移除，但保留文档记录，便于随后原地更新。"""
        doc = self.state["documents"].get(doc_id)
        if not doc:
            return
        cid = doc.get("cluster")
        if cid:
            self._remove_from_cluster(doc_id, cid)
        self._apply_df_delta(doc.get("term_counts", {}), -1)

    def _remove_document(self, doc_id: str) -> None:
        doc = self.state["documents"].pop(doc_id, None)
        if not doc:
            return
        cid = doc.get("cluster")
        if cid:
            self._remove_from_cluster(doc_id, cid)

    def _normalize_cluster_center(self, cid: str) -> None:
        cluster = self.state["clusters"].get(cid)
        if not cluster:
            return
        if cluster["size"] > 0:
            mean = {term: value / cluster["size"]
                    for term, value in cluster["sum_vector"].items()}
            cluster["centroid"] = _round_vec(normalize_sparse(mean))
        else:
            cluster["centroid"] = {}

    def _reset_cluster_stats(self, cid: str) -> None:
        cluster = self.state["clusters"][cid]
        cluster.update({
            "centroid": {}, "sum_vector": {}, "size": 0,
            "term_freq": {}, "term_doc_freq": {},
        })

    def _assign_to_cluster(self, doc_id: str, cid: str, vec: dict,
                           center: dict, save_score: bool = True) -> None:
        doc = self.state["documents"][doc_id]
        doc["vector"] = _round_vec(vec)
        doc["cluster"] = cid
        if save_score:
            doc["score"] = round(cosine_sparse(vec, center), 6)
        self._add_to_cluster(doc_id, cid, doc)

    def _remove_empty_clusters(self) -> None:
        for cid in [cid for cid, c in self.state["clusters"].items()
                    if c.get("size", 0) <= 0]:
            self.state["clusters"].pop(cid, None)

    def _refresh_old_vectors(self, remove_empty: bool = True) -> None:
        """复用缓存词频，按更新后的 IDF 重算已有簇中文档的向量。

        新增文档此时尚未分配簇，不参与拆分判断；这样“新主题出现”会在它落簇后
        通过局部修正处理，而不会污染旧簇拆分。
        """
        docs = self.state["documents"]
        model = TfidfModel(self.state["document_frequency"], len(docs))
        for doc in docs.values():
            if doc.get("cluster") is not None:
                doc["vector"] = _round_vec(model.vectorize(doc.get("term_counts", {})))
        for cid, cluster in self.state["clusters"].items():
            cluster["sum_vector"] = {}
            cluster["term_freq"] = {}
            cluster["term_doc_freq"] = {}
            cluster["size"] = 0
        for doc_id, doc in docs.items():
            cid = doc.get("cluster")
            if cid:
                self._add_to_cluster(doc_id, cid, doc)
        if remove_empty:
            self._remove_empty_clusters()

    # -- 簇数调整 ---------------------------------------------------------
    def _adjust_to_fixed_k(self, target_k: int,
                           previous_centroids: Optional[dict] = None) -> None:
        n_docs = len(self.state["documents"])
        target = max(1, min(target_k, n_docs))
        guard = 0
        while len(self.state["clusters"]) < target and guard < target + 2:
            if not self._split_best_cluster(force=True,
                                            previous_centroids=previous_centroids):
                break
            guard += 1
        while len(self.state["clusters"]) > target:
            if not self._merge_closest_pair():
                break

    def _auto_adjust_clusters(self, max_topics: int,
                              previous_centroids: Optional[dict] = None) -> None:
        if not self.state["documents"]:
            self.state["clusters"] = {}
            return
        if not self.state["clusters"]:
            self._create_cluster()
        # 超过上限先合并最相近的簇。
        while len(self.state["clusters"]) > max_topics:
            if not self._merge_closest_pair():
                break
        # 小而近的簇合并；大而散的簇拆分。每轮最多数次，避免更新抖动。
        for _ in range(3):
            if self._merge_small_nearby():
                continue
            if len(self.state["clusters"]) < max_topics and self._split_best_cluster(
                    previous_centroids=previous_centroids):
                continue
            break

    def _split_best_cluster(self, force: bool = False,
                            previous_centroids: Optional[dict] = None) -> bool:
        candidates = []
        n_docs = len(self.state["documents"])
        k = max(1, len(self.state["clusters"]))
        avg = n_docs / k
        for cid, cluster in self.state["clusters"].items():
            member_ids = [doc_id for doc_id, doc in self.state["documents"].items()
                          if doc.get("cluster") == cid]
            if len(member_ids) < _MIN_SPLIT_SIZE:
                continue
            vectors = [self.state["documents"][i]["vector"] for i in member_ids]
            labels, centers = two_means(vectors)
            if any(sum(1 for label in labels if label == part) < 2 for part in (0, 1)):
                continue
            old_center = (previous_centroids or {}).get(cid, cluster.get("centroid", {}))
            old_score = sum(cosine_sparse(vec, old_center) for vec in vectors) / len(vectors)
            new_score = sum(cosine_sparse(vec, centers[label])
                            for vec, label in zip(vectors, labels)) / len(vectors)
            separation = cosine_sparse(centers[0], centers[1])
            gain = new_score - old_score
            size_bonus = len(member_ids) / max(avg, 1) if k > 1 else 1.0
            # 强制拆分是为满足用户指定 k；自动拆分必须看到实质增益。
            # previous_centroids 缺失时表示完整/首次场景，可用当前质心。
            is_new_split = bool(previous_centroids) and cid not in previous_centroids
            if (force or (
                    gain > 0.015 and separation < 0.82 and
                    (k == 1 or size_bonus >= 1.1) and not is_new_split)):
                candidates.append((gain, len(member_ids), cid, member_ids, labels, centers))
        if not candidates:
            return False
        _, _, cid, member_ids, labels, centers = max(
            candidates, key=lambda item: (item[0], item[1], item[2]))
        identity_center = (previous_centroids or {}).get(cid, cluster.get("centroid", {}))
        new_id = self._create_cluster()
        self._reset_cluster_stats(cid)
        self._reset_cluster_stats(new_id)
        # 让与旧质心更接近的一半保留原簇号，另一半使用新簇号，增强稳定性。
        if cosine_sparse(identity_center, centers[1]) > cosine_sparse(
                identity_center, centers[0]):
            target_ids = [new_id, cid]
        else:
            target_ids = [cid, new_id]
        for doc_id, label, vec in zip(member_ids, labels,
                                      [self.state["documents"][i]["vector"]
                                       for i in member_ids]):
            target = target_ids[label]
            doc = self.state["documents"][doc_id]
            doc["cluster"] = target
            doc["score"] = round(cosine_sparse(vec, centers[label]), 6)
            self._add_to_cluster(doc_id, target, doc)
        return True

    def _merge_small_nearby(self) -> bool:
        ids = list(self.state["clusters"])
        if len(ids) <= 1:
            return False
        n_docs = len(self.state["documents"])
        avg = n_docs / len(ids)
        best = None
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                ca, cb = self.state["clusters"][a], self.state["clusters"][b]
                sim = cosine_sparse(ca.get("centroid", {}), cb.get("centroid", {}))
                smaller = min(ca["size"], cb["size"])
                # 近邻强合并；极小簇即使不太近也合并，防止一篇文档长期成簇。
                near = sim > 0.32
                tiny = smaller <= max(2, int(avg / 4))
                singleton = smaller <= 1 and n_docs > len(ids)
                if near or tiny or singleton:
                    score = sim + (0.2 if tiny else 0.0) + (0.1 if singleton else 0.0)
                    if best is None or score > best[0]:
                        best = (score, a, b)
        if not best:
            return False
        self._merge_clusters(best[1], best[2])
        return True

    def _merge_closest_pair(self) -> bool:
        ids = list(self.state["clusters"])
        if len(ids) <= 1:
            return False
        best = None
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                sim = cosine_sparse(self.state["clusters"][a].get("centroid", {}),
                                    self.state["clusters"][b].get("centroid", {}))
                if best is None or sim > best[0]:
                    best = (sim, a, b)
        if not best:
            return False
        self._merge_clusters(best[1], best[2])
        return True

    def _merge_clusters(self, keep_id: str, remove_id: str) -> None:
        member_ids = [doc_id for doc_id, doc in self.state["documents"].items()
                      if doc.get("cluster") == remove_id]
        for doc_id in member_ids:
            doc = self.state["documents"][doc_id]
            doc["cluster"] = keep_id
            self._add_to_cluster(doc_id, keep_id, doc)
        self.state["clusters"].pop(remove_id, None)
        self._normalize_cluster_center(keep_id)

    def _auto_merge_remote_singletons(self) -> None:
        # 局部 k-means 可能留下单文档簇；若存在其它簇则并入最近簇。
        while len(self.state["clusters"]) > 1:
            singleton = next((cid for cid, c in self.state["clusters"].items()
                              if c.get("size", 0) <= 1), None)
            if not singleton:
                return
            others = [cid for cid in self.state["clusters"] if cid != singleton]
            center = self.state["clusters"][singleton].get("centroid", {})
            nearest = max(others,
                          key=lambda cid: cosine_sparse(
                              center, self.state["clusters"][cid].get("centroid", {})))
            self._merge_clusters(nearest, singleton)

    def _refine_clusters(self, cluster_ids: set[str]) -> None:
        ids = [cid for cid in self._ordered_cluster_ids() if cid in cluster_ids]
        if len(ids) < 1:
            return
        member_ids = [doc_id for doc_id, doc in self.state["documents"].items()
                      if doc.get("cluster") in ids]
        if not member_ids:
            return
        vectors = [self.state["documents"][doc_id]["vector"] for doc_id in member_ids]
        if len(ids) == 1:
            center = average_vectors(vectors)
            for doc_id, vec in zip(member_ids, vectors):
                doc = self.state["documents"][doc_id]
                doc["score"] = round(cosine_sparse(vec, center), 6)
            cid = ids[0]
            self._reset_cluster_stats(cid)
            for doc_id in member_ids:
                self._add_to_cluster(doc_id, cid, self.state["documents"][doc_id])
            return
        initial = [self.state["clusters"][cid].get("centroid", {}) for cid in ids]
        labels, centers = spherical_kmeans(
            vectors, len(ids), initial_centers=initial)
        for cid in ids:
            self._reset_cluster_stats(cid)
        for doc_id, label, vec in zip(member_ids, labels, vectors):
            cid = ids[label]
            doc = self.state["documents"][doc_id]
            doc["cluster"] = cid
            doc["score"] = round(cosine_sparse(vec, centers[label]), 6)
            self._add_to_cluster(doc_id, cid, doc)

    def _reassign_changed_documents(self, changed_ids: list[str]) -> None:
        for doc_id in changed_ids:
            doc = self.state["documents"][doc_id]
            old_cid = doc.get("cluster")
            new_cid, score, second = self._nearest_cluster(doc["vector"])
            if new_cid and new_cid != old_cid:
                if old_cid:
                    self._remove_from_cluster(doc_id, old_cid)
                doc["cluster"] = new_cid
                self._add_to_cluster(doc_id, new_cid, doc)
            elif new_cid:
                doc["score"] = score
            doc["ambiguity"] = round(second - score, 6)

    # -- 标签与代表文档 ---------------------------------------------------
    def _refresh_all_cluster_metadata(self) -> None:
        for cid in list(self.state["clusters"]):
            self._refresh_cluster_metadata(cid)

    def _refresh_changed_metadata(self, cluster_ids: set[str]) -> None:
        for cid in cluster_ids:
            if cid in self.state["clusters"]:
                self._refresh_cluster_metadata(cid)

    def _refresh_cluster_metadata(self, cid: str) -> None:
        cluster = self.state["clusters"].get(cid)
        if not cluster:
            return
        df = self.state["document_frequency"]
        n_docs = max(1, len(self.state["documents"]))
        size = max(1, cluster["size"])
        scored = []
        for term, freq in cluster.get("term_freq", {}).items():
            idf = math.log((n_docs + 1) / (df.get(term, 0) + 1)) + 1.0
            cluster_presence = cluster.get("term_doc_freq", {}).get(term, 0) / size
            score = (freq / size) * idf * (0.65 + 0.35 * math.sqrt(cluster_presence))
            scored.append({"term": term, "score": round(score, 6),
                           "cluster_df": cluster.get("term_doc_freq", {}).get(term, 0),
                           "df": df.get(term, 0)})
        scored.sort(key=lambda item: (-item["score"], item["term"]))
        keywords = scored[:10]
        cluster["keywords"] = keywords
        cluster["label"] = make_label(keywords)
        members = [self.state["documents"][doc_id]
                   for doc_id, doc in self.state["documents"].items()
                   if doc.get("cluster") == cid]
        if members and cluster.get("centroid"):
            cohesion = sum(cosine_sparse(doc["vector"], cluster["centroid"])
                           for doc in members) / len(members)
        else:
            cohesion = 0.0
        cluster["cohesion"] = round(cohesion, 6)

    def _nearest_cluster(self, vec: dict) -> tuple[Optional[str], float, float]:
        ids = self._ordered_cluster_ids()
        if not ids:
            return None, 0.0, 0.0
        scored = [(cosine_sparse(vec, self.state["clusters"][cid].get("centroid", {})), cid)
                  for cid in ids]
        scored.sort(key=lambda item: (-item[0], item[1]))
        best_sim, best_id = scored[0]
        second = scored[1][0] if len(scored) > 1 else best_sim
        return best_id, round(max(0.0, best_sim), 6), round(max(0.0, second), 6)

    def _stabilize_cluster_ids(self, temp_ids: list[str]) -> None:
        """用语料中最小文档 ID 排序簇，再映射成稳定的 topic_1、topic_2..."""
        clusters = self.state["clusters"]
        def first_doc(temp_id: str) -> str:
            members = [doc_id for doc_id, doc in self.state["documents"].items()
                       if doc.get("cluster") == temp_id]
            return min(members, default="zzzz")
        ordered_temp = sorted(temp_ids, key=first_doc)
        mapping = {temp: f"topic_{i + 1}" for i, temp in enumerate(ordered_temp)}
        for doc in self.state["documents"].values():
            if doc.get("cluster") in mapping:
                doc["cluster"] = mapping[doc["cluster"]]
        rebuilt = {mapping[temp]: clusters[temp] for temp in ordered_temp
                   if temp in clusters}
        self.state["clusters"] = rebuilt

    def _canonicalize_cluster_ids(self) -> None:
        """把增量产生的随机簇号整理成 topic_1...，同时尽量保留旧编号。"""
        clusters = self.state.get("clusters", {})
        if not clusters:
            return

        def first_doc(cid: str) -> str:
            return min((doc_id for doc_id, doc in self.state["documents"].items()
                        if doc.get("cluster") == cid), default="~")

        old_ids = list(clusters)
        existing = {cid for cid in old_ids if cid.startswith("topic_")}
        # 已有 topic_N 先保留自己的编号，避免文档删除后簇号整体漂移；新簇补空位。
        mapping = {cid: cid for cid in existing}
        used = set(existing)
        next_number = 1
        for cid in sorted(old_ids, key=lambda cid: (first_doc(cid), cid)):
            if cid in mapping:
                continue
            while f"topic_{next_number}" in used:
                next_number += 1
            mapping[cid] = f"topic_{next_number}"
            used.add(mapping[cid])

        if all(old == new for old, new in mapping.items()):
            return
        for doc in self.state["documents"].values():
            if doc.get("cluster") in mapping:
                doc["cluster"] = mapping[doc["cluster"]]
        self.state["clusters"] = {
            mapping[cid]: clusters[cid] for cid in old_ids if cid in clusters
        }

    def _ordered_cluster_ids(self) -> list[str]:
        return sorted(self.state.get("clusters", {}),
                      key=lambda cid: (not cid.startswith("topic_"), cid))

    def _members(self, cluster_id: str) -> list[dict]:
        return [doc for doc in self.state["documents"].values()
                if doc.get("cluster") == cluster_id]

    def _doc_brief(self, doc: dict, preview: bool = False) -> dict:
        # 代表文档详情由路由从语料库补全文，避免模型状态重复保存大文本。
        name = doc.get("name") or doc["id"]
        brief = {
            "id": doc["id"],
            "name": name,
            "score": round(doc.get("score", 0.0), 4),
            "ambiguity": round(doc.get("ambiguity", 0.0), 4),
            "length": doc.get("length", 0),
            "shard": doc.get("shard"),
            "created_at": doc.get("created_at", 0),
        }
        if preview:
            # 仅保存短预览；全文仍可从语料库接口查看。
            brief["preview"] = doc.get("preview", "")
        return brief

    def _distance_matrix(self, ids: list[str], centers: dict) -> list[list[float]]:
        matrix = []
        for i, a in enumerate(ids):
            row = []
            for j, b in enumerate(ids):
                if i == j:
                    row.append(0.0)
                else:
                    row.append(max(0.0, 1.0 - cosine_sparse(centers[a], centers[b])))
            matrix.append(row)
        return matrix

    def _current_fingerprints(self) -> dict:
        return {str(item["index"]): item
                for item in self.corpus.shard_fingerprints()}

    @staticmethod
    def _normalize_requested(value) -> Optional[int]:
        if value in (None, "", 0, "0", "auto"):
            return None
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None
        return max(1, number)

    # -- 持久化 -----------------------------------------------------------
    def _save(self) -> None:
        _atomic_write_json(self.state_path, self.state)
