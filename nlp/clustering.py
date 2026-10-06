"""文档话题聚类。

纯 Python 实现（无 numpy），面向「语料分片存放、增量更新、结果稳定」的场景：

1. **向量化**：分词去停用词后取长度 ≥ 2 的词（单字多为分词碎片，易造成
   跨话题链式粘连），构建「词 TF-IDF + 词内字符二元组 TF-IDF」加权混合的
   单位稀疏向量。二元组特征可桥接分词不一致（如「足球联赛」与「足球」），
   提升短文本下同话题文档的相似度。
2. **聚类**：先做一次**平均连接凝聚层次聚类**（自底向上合并，正交文档只在
   被迫达到目标 k 时才合并），再以凝聚结果为初始中心做球面 k-means 精修。
   全程无随机数；文档遍历、并列取舍均按 id 字典序 —— 同一批文档多次聚类
   结果完全一致，不会「每次簇都变样」。
3. **话题数**：可由用户指定；也可自动估计 —— 候选 k 范围
   ``2 .. min(max_k, n-1)``（``max_k`` 缺省取 √n 的话题粒度先验），以轮廓
   系数（silhouette）最优者为准（并列取较小 k；单文档簇轮廓记 0，避免
   自动估计退化为「每篇一簇」）。文档多寡、话题界限模糊时也能给出合理
   划分，并为每篇文档给出置信度与「边界模糊」标记。
4. **标签与关键词**：c-TF-IDF（簇内词频 × 全局 IDF）取头部词，保证标签词
   一定真实出现在簇内文档中；标签取前两个关键词。
5. **代表文档**：按与簇中心的余弦相似度排序（medoid 式）。
6. **簇间距离与地图**：簇中心余弦距离矩阵 + 经典 MDS 二维投影。
7. **增量更新**：:class:`ClusterIndex` 缓存每篇文档的内容哈希与词频；新文档
   入库 / 旧文档删改时，只有受影响文档重新分词，未变更文档直接复用缓存
   词频；若语料与参数完全未变（指纹一致），直接复用上次聚类结果。

复杂度：凝聚层次聚类为 O(n^3)（n 为文档数），适合数百篇规模的语料；
更大语料建议先分片抽样或提高 shard 内聚类再合并。
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from typing import Optional

from .embeddings import _eigh_power
from .lexicon import STOPWORDS
from .segmenter import Segmenter


# 与次优簇中心的相似度差小于该值时，认为文档处于话题边界
_AMBIGUOUS_MARGIN = 0.02

# 混合向量中词特征的权重（其余为字符二元组特征权重）
_WORD_WEIGHT = 0.65


def _content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _dot(a: dict, b: dict) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def _normalize(vec: dict) -> dict:
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm <= 0:
        return dict(vec)
    return {k: v / norm for k, v in vec.items()}


class DocumentClusterer:
    """确定性文档聚类器（凝聚初始化 + 球面 k-means + 自动 k + c-TF-IDF 标签）。"""

    def __init__(self, segmenter: Optional[Segmenter] = None,
                 word_weight: float = _WORD_WEIGHT):
        self.segmenter = segmenter or Segmenter()
        self.word_weight = word_weight

    # -- 分词 -------------------------------------------------------------
    def tokenize(self, text: str) -> dict[str, int]:
        """分词、去停用词、丢弃单字碎片，返回 ``{词: 频次}``。"""
        counts: Counter = Counter()
        for w in self.segmenter.cut(text):
            if w in STOPWORDS or not w.strip():
                continue
            if len(w) < 2:
                # 单字多为分词碎片（专名 / 数字 / 虚词残余），会桥接无关文档
                continue
            counts[w] += 1
        return dict(counts)

    @staticmethod
    def _bigrams(terms: dict[str, int]) -> dict[str, int]:
        """词内字符二元组特征（单字词以其自身充当）。"""
        bg: Counter = Counter()
        for w, cnt in terms.items():
            if len(w) == 1:
                bg[w] += cnt
            else:
                for i in range(len(w) - 1):
                    bg[w[i:i + 2]] += cnt
        return dict(bg)

    # -- 向量化 -----------------------------------------------------------
    @staticmethod
    def _idf(doc_terms: dict[str, dict[str, int]]) -> dict[str, float]:
        n = len(doc_terms)
        df: Counter = Counter()
        for terms in doc_terms.values():
            for w in terms:
                df[w] += 1
        return {w: math.log((n + 1) / (c + 1)) + 1.0 for w, c in df.items()}

    @staticmethod
    def _vectorize(terms: dict[str, int], idf: dict[str, float]) -> dict[str, float]:
        total = sum(terms.values()) or 1
        vec = {w: (c / total) * idf.get(w, 0.0) for w, c in terms.items()}
        return _normalize(vec)

    def _hybrid_vectors(self, doc_terms: dict[str, dict[str, int]]) -> dict:
        """词 TF-IDF 与字符二元组 TF-IDF 各自归一化后按权重拼接。

        拼接向量的点积 = word_weight·cos_词 + (1-word_weight)·cos_二元组。
        """
        word_idf = self._idf(doc_terms)
        bigram_terms = {d: self._bigrams(t) for d, t in doc_terms.items()}
        bigram_idf = self._idf(bigram_terms)
        wa = math.sqrt(self.word_weight)
        ba = math.sqrt(1.0 - self.word_weight)
        vectors = {}
        for d, terms in doc_terms.items():
            vec = {}
            for w, x in self._vectorize(terms, word_idf).items():
                vec[("w", w)] = x * wa
            for b, x in self._vectorize(bigram_terms[d], bigram_idf).items():
                vec[("b", b)] = x * ba
            vectors[d] = vec
        return vectors

    # -- 聚类主流程 -------------------------------------------------------
    def cluster(self, doc_terms: dict[str, dict[str, int]],
                n_clusters: Optional[int] = None,
                max_k: Optional[int] = None,
                n_keywords: int = 8,
                n_representatives: int = 5,
                max_iter: int = 100) -> dict:
        """对 ``{文档id: {词: 频次}}`` 聚类，返回完整结果 dict（可 JSON 序列化）。"""
        # 无有效词的文档不参与聚类，单独列出
        valid = {d: t for d, t in doc_terms.items() if t}
        excluded = sorted(d for d in doc_terms if d not in valid)
        doc_ids = sorted(valid)
        n = len(doc_ids)
        if n == 0:
            raise ValueError("没有可聚类的文档（文档为空或全是停用词/单字）")

        vectors = self._hybrid_vectors(valid)
        # 余弦相似度矩阵（单位向量点积）
        sim = [[_dot(vectors[doc_ids[i]], vectors[doc_ids[j]])
                for j in range(n)] for i in range(n)]

        if n == 1:
            labels, centers = {doc_ids[0]: 0}, [vectors[doc_ids[0]]]
            best_score, k_scores, k = 0.0, [], 1
        else:
            if n_clusters:
                candidates = [max(1, min(int(n_clusters), n))]
            elif n == 2:
                candidates = [2]
            else:
                cap = int(max_k) if max_k else max(2, round(n ** 0.5))
                cap = max(2, min(cap, n - 1))
                candidates = list(range(2, cap + 1))

            # 一次凝聚层次聚类，记录每个候选 k 的划分作为 k-means 初始划分
            snapshots = (self._agglomerative(doc_ids, sim, min(candidates))
                         if n > 2 else {})

            best = None
            k_scores = []
            for k in candidates:
                if k >= n:
                    labels0 = {d: i for i, d in enumerate(doc_ids)}
                elif k == 1:
                    labels0 = {d: 0 for d in doc_ids}
                else:
                    labels0 = snapshots[k]
                labels_k, centers_k = self._kmeans(
                    doc_ids, vectors, k, labels0, max_iter)
                score = self._silhouette(doc_ids, sim, labels_k, k)
                k_scores.append({"k": k, "score": round(score, 4)})
                # 并列时保留较小的 k（candidates 升序，严格大于才替换）
                if best is None or score > best[0] + 1e-12:
                    best = (score, labels_k, centers_k, k)
            best_score, labels, centers, k = best

        result = self._build_result(doc_ids, valid, vectors, labels, centers,
                                    k, n_keywords, n_representatives)
        result.update({
            "n_docs": len(doc_terms),
            "n_clustered": n,
            "excluded": excluded,
            "auto_k": not bool(n_clusters),
            "k_scores": k_scores,
            "silhouette": round(best_score, 4),
        })
        return result

    # -- 凝聚层次聚类（平均连接） -----------------------------------------
    @staticmethod
    def _agglomerative(doc_ids, sim, k_min) -> dict[int, dict]:
        """平均连接凝聚聚类，返回 ``{k: 划分}``（k 从 n-1 到 max(2, k_min)）。

        簇间相似度 = 跨簇文档对的平均相似度（Lance-Williams 和式维护）；
        并列时取簇内最小文档 id 字典序较小的一对，保证确定性。
        """
        n = len(doc_ids)
        members = [[i] for i in range(n)]
        csum = [row[:] for row in sim]
        sizes = [1] * n
        active = list(range(n))
        snaps: dict[int, dict] = {}

        while len(active) > max(2, k_min):
            best = None
            for x in range(len(active)):
                a = active[x]
                for y in range(x + 1, len(active)):
                    b = active[y]
                    s = csum[a][b] / (sizes[a] * sizes[b])
                    key = (-s, a, b)
                    if best is None or key < best[0]:
                        best = (key, a, b)
            _, a, b = best
            for c in active:
                if c == a or c == b:
                    continue
                csum[a][c] = csum[c][a] = csum[a][c] + csum[b][c]
            sizes[a] += sizes[b]
            members[a] = members[a] + members[b]
            active.remove(b)

            labels = {}
            ordered = sorted(active, key=lambda cl: min(members[cl]))
            for rank, cl in enumerate(ordered):
                for i in members[cl]:
                    labels[doc_ids[i]] = rank
            snaps[len(active)] = labels
        return snaps

    # -- 球面 k-means（从给定初始划分精修） --------------------------------
    def _kmeans(self, doc_ids, vectors, k, labels0, max_iter):
        centers = self._centroids(doc_ids, vectors, labels0, k)
        labels = dict(labels0)
        for _ in range(max_iter):
            # 分配：并列取编号最小的簇
            new_labels = {}
            for d in doc_ids:
                best_c, best_s = 0, -2.0
                for c in range(k):
                    s = _dot(vectors[d], centers[c])
                    if s > best_s + 1e-12:
                        best_c, best_s = c, s
                new_labels[d] = best_c
            if new_labels == labels:
                break
            labels = new_labels
            centers = self._centroids(doc_ids, vectors, labels, k)
        return labels, centers

    def _centroids(self, doc_ids, vectors, labels, k):
        """各簇成员单位向量均值再归一化；空簇用最不贴合的文档补位。"""
        sums = [dict() for _ in range(k)]
        counts = [0] * k
        for d in doc_ids:
            c = labels[d]
            counts[c] += 1
            for key, v in vectors[d].items():
                sums[c][key] = sums[c].get(key, 0.0) + v
        centers = []
        for c in range(k):
            if counts[c]:
                centers.append(_normalize(sums[c]))
            else:
                donor = self._worst_fit_doc(doc_ids, vectors, labels, sums, counts)
                centers.append(dict(vectors[donor]))
        return centers

    @staticmethod
    def _worst_fit_doc(doc_ids, vectors, labels, sums, counts):
        """与自身簇中心相似度最低的文档（所在簇须至少有 2 篇）。"""
        worst_d, worst_s = None, None
        for d in doc_ids:
            c = labels[d]
            if counts[c] <= 1:
                continue
            s = _dot(vectors[d], _normalize(sums[c]))
            if worst_s is None or s < worst_s - 1e-12 or (
                    abs(s - worst_s) <= 1e-12 and d < worst_d):
                worst_d, worst_s = d, s
        return worst_d if worst_d is not None else doc_ids[0]

    # -- 轮廓系数 ---------------------------------------------------------
    @staticmethod
    def _silhouette(doc_ids, sim, labels, k) -> float:
        n = len(doc_ids)
        if k <= 1 or n <= k:
            return 0.0
        clusters: dict[int, list[int]] = {}
        index = {d: i for i, d in enumerate(doc_ids)}
        for d in doc_ids:
            clusters.setdefault(labels[d], []).append(index[d])

        total = 0.0
        for d in doc_ids:
            i = index[d]
            own = clusters[labels[d]]
            if len(own) <= 1:
                # 单文档簇按惯例记 0，避免自动 k 退化为每篇一簇
                continue
            a = sum(1.0 - sim[i][j] for j in own if j != i) / (len(own) - 1)
            b = None
            for c, members in clusters.items():
                if c == labels[d]:
                    continue
                dist = sum(1.0 - sim[i][j] for j in members) / len(members)
                if b is None or dist < b:
                    b = dist
            if b is None:
                continue
            denom = max(a, b)
            if denom > 0:
                total += (b - a) / denom
        return total / n

    # -- 结果组装 ---------------------------------------------------------
    def _build_result(self, doc_ids, doc_terms, vectors, labels, centers,
                      k, n_keywords, n_representatives):
        idf = self._idf(doc_terms)
        clusters = []
        assignments: dict[str, dict] = {}

        for c in range(k):
            members = [d for d in doc_ids if labels[d] == c]
            # c-TF-IDF：簇内词频 × 全局 IDF，保证关键词真实出自簇内
            ctf: Counter = Counter()
            for d in members:
                ctf.update(doc_terms[d])
            scored = sorted(
                ((w, tf * idf.get(w, 0.0)) for w, tf in ctf.items()),
                key=lambda x: (-x[1], x[0]))
            keywords = [{"word": w, "score": round(s, 4)}
                        for w, s in scored[:n_keywords] if s > 0]
            label = " · ".join(kw["word"] for kw in keywords[:2]) or f"话题 {c + 1}"

            center = centers[c]
            ranked = sorted(members,
                            key=lambda d: (-_dot(vectors[d], center), d))
            cohesion = (sum(_dot(vectors[d], center) for d in members)
                        / len(members)) if members else 0.0
            clusters.append({
                "id": c,
                "label": label,
                "keywords": keywords,
                "size": len(members),
                "cohesion": round(cohesion, 4),
                "representatives": [
                    {"doc_id": d, "score": round(_dot(vectors[d], center), 4)}
                    for d in ranked[:n_representatives]],
            })

            for d in members:
                sim_own = _dot(vectors[d], center)
                sim_other = max(
                    (_dot(vectors[d], centers[o]) for o in range(k) if o != c),
                    default=-1.0)
                confidence = (sim_own - sim_other + 1.0) / 2.0
                assignments[d] = {
                    "cluster": c,
                    "score": round(sim_own, 4),
                    "confidence": round(min(max(confidence, 0.0), 1.0), 4),
                    "ambiguous": bool(k > 1 and
                                      (sim_own - sim_other) < _AMBIGUOUS_MARGIN),
                }

        # 簇间距离矩阵（簇中心余弦距离）
        distances = []
        for i in range(k):
            row = []
            for j in range(k):
                row.append(round(1.0 - _dot(centers[i], centers[j]), 4))
            distances.append(row)

        # 簇中心 MDS 二维投影（话题地图坐标）
        points = self._mds_2d(distances)
        for c in range(k):
            clusters[c]["center"] = [round(points[c][0], 4),
                                     round(points[c][1], 4)]

        return {
            "n_clusters": k,
            "clusters": clusters,
            "assignments": assignments,
            "distances": distances,
        }

    # -- 经典 MDS ---------------------------------------------------------
    @staticmethod
    def _mds_2d(dist: list[list[float]]) -> list[list[float]]:
        n = len(dist)
        if n == 0:
            return []
        if n == 1:
            return [[0.0, 0.0]]
        d2 = [[dist[i][j] ** 2 for j in range(n)] for i in range(n)]
        row_mean = [sum(r) / n for r in d2]
        total_mean = sum(row_mean) / n
        # 双中心化：B = -1/2 · J D² J（方阵行列均值相同）
        B = [[-0.5 * (d2[i][j] - row_mean[i] - row_mean[j] + total_mean)
              for j in range(n)] for i in range(n)]
        vecs = _eigh_power(B, 2)
        coords = [[0.0, 0.0] for _ in range(n)]
        for axis, v in enumerate(vecs[:2]):
            lam = sum(v[i] * sum(B[i][j] * v[j] for j in range(n))
                      for i in range(n))
            scale = math.sqrt(max(lam, 0.0))
            # 符号规范化：绝对值最大的分量取正，保证结果可复现
            if v:
                pivot = max(range(n), key=lambda i: abs(v[i]))
                sign = 1.0 if v[pivot] >= 0 else -1.0
            else:
                sign = 1.0
            for i in range(n):
                coords[i][axis] = v[i] * scale * sign
        return coords


class ClusterIndex:
    """增量聚类索引：缓存文档词频，语料增删改时只重算受影响的部分。

    状态为纯 dict（``state``），由调用方负责持久化（本平台存为
    ``data/models/cluster_state.json``）。工作流程：

    1. 对每篇文档计算内容哈希，与缓存比对，得到 新增 / 变更 / 删除 集合；
    2. 仅对新增与变更文档重新分词，未变更文档复用缓存词频；
    3. 若「语料指纹 + 聚类参数」与上次完全一致，直接复用上次结果；
    4. 否则基于缓存词频重算 IDF / 向量并执行确定性聚类。
    """

    STATE_VERSION = 1

    def __init__(self, clusterer: Optional[DocumentClusterer] = None,
                 state: Optional[dict] = None):
        self.clusterer = clusterer or DocumentClusterer()
        self.state = state or {
            "version": self.STATE_VERSION,
            "docs": {},          # doc_id -> {"hash": str, "terms": {词: 频次}}
            "params": None,
            "fingerprint": None,
            "result": None,
        }

    # -- 指纹 -------------------------------------------------------------
    @staticmethod
    def fingerprint(doc_hashes: dict[str, str], params: dict) -> str:
        h = hashlib.sha1()
        h.update(json.dumps(params, sort_keys=True,
                            ensure_ascii=False).encode("utf-8"))
        for doc_id in sorted(doc_hashes):
            h.update(f"{doc_id}:{doc_hashes[doc_id]};".encode("utf-8"))
        return h.hexdigest()[:16]

    # -- 增量更新 ---------------------------------------------------------
    def update(self, documents: dict[str, str], params: Optional[dict] = None) -> dict:
        """用最新语料 ``{doc_id: text}`` 增量更新索引并按需重算聚类。

        返回 ``{"result": ..., "reused": bool, "updates": {...}}``。
        """
        params = {
            "n_clusters": None, "max_k": None,
            "n_keywords": 8, "n_representatives": 5,
            **(params or {}),
        }
        cached: dict = self.state["docs"]
        new_hashes = {d: _content_hash(t) for d, t in documents.items()}

        added, changed, removed = [], [], []
        for d in sorted(new_hashes):
            if d not in cached:
                added.append(d)
            elif cached[d]["hash"] != new_hashes[d]:
                changed.append(d)
        for d in sorted(cached):
            if d not in new_hashes:
                removed.append(d)

        for d in removed:
            del cached[d]
        for d in added + changed:
            cached[d] = {
                "hash": new_hashes[d],
                "terms": self.clusterer.tokenize(documents[d]),
            }

        updates = {
            "added": len(added), "changed": len(changed),
            "removed": len(removed), "vectorized": len(added) + len(changed),
            "cached": len(cached),
        }
        fingerprint = self.fingerprint(new_hashes, params)
        if (fingerprint == self.state.get("fingerprint")
                and self.state.get("result") is not None):
            return {"result": self.state["result"], "reused": True,
                    "updates": {**updates, "vectorized": 0}}

        doc_terms = {d: cached[d]["terms"] for d in sorted(cached)}
        result = self.clusterer.cluster(doc_terms, **params)
        result["fingerprint"] = fingerprint
        result["created_at"] = time.time()
        self.state.update({"params": params, "fingerprint": fingerprint,
                           "result": result})
        return {"result": result, "reused": False, "updates": updates}
