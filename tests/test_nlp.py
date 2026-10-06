"""NLP 平台单元测试。

运行：``python -m unittest discover -s tests -v``
覆盖：分词、词性、句法、NER、情感、摘要、翻译、关键词、词向量、
分片存储（含并发锁）、流水线引擎、HMM。
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import unittest

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp import (get_segmenter, get_tagger, get_parser, get_constituency_parser,
                 get_ner, get_sentiment, get_summarizer, get_translator,
                 get_keywords, get_embeddings, TAGSET,
                 DocumentClusterer, ClusterIndex)
from nlp.hmm import HMM
from pipeline import PipelineEngine, PipelineError
from storage import ShardedStore, StoreRegistry


class TestSegmenter(unittest.TestCase):
    def test_basic(self):
        words = get_segmenter().cut("自然语言处理是人工智能的重要分支")
        self.assertIn("自然语言", words)
        self.assertIn("人工智能", words)
        self.assertIn("是", words)

    def test_english_number(self):
        words = get_segmenter().cut("我用Python写了100行代码")
        self.assertIn("Python", words)
        self.assertIn("100", words)


class TestPOSTagger(unittest.TestCase):
    def test_tags(self):
        pairs = get_tagger().tag("我学习自然语言处理")
        self.assertTrue(pairs)
        for word, tag in pairs:
            self.assertIn(tag, TAGSET, f"{word}:{tag}")

    def test_punct_as_other(self):
        pairs = get_tagger().tag("你好，世界。")
        tags = [t for _, t in pairs]
        for t in tags:
            if t in ("，", "。"):
                continue
        # 标点词本身应为 x
        for w, t in pairs:
            if w in ("，", "。"):
                self.assertEqual(t, "x")


class TestParser(unittest.TestCase):
    def test_dependency(self):
        dep = get_parser().parse("北京大学的研究团队开发了机器学习系统")
        self.assertEqual(len(dep["words"]), len(dep["heads"]))
        self.assertIn(-1, dep["heads"])  # 存在根
        # 每个 head 都是有效下标或 -1
        for h in dep["heads"]:
            self.assertTrue(h == -1 or 0 <= h < len(dep["words"]))

    def test_constituency_spans(self):
        c = get_constituency_parser().parse("北京大学的研究团队开发了系统")
        leaves = self._leaves(c["tree"])
        self.assertEqual("北京大学的研究团队开发了系统", leaves)

    @staticmethod
    def _leaves(tree):
        if not tree.get("children"):
            return tree.get("word", "")
        return "".join(TestParser._leaves(ch) for ch in tree["children"])


class TestNER(unittest.TestCase):
    def test_known_entities(self):
        ents = get_ner().recognize("马云在北京工作")
        types = {e["text"]: e["type"] for e in ents}
        self.assertEqual(types.get("马云"), "PERSON")
        self.assertEqual(types.get("北京"), "LOCATION")

    def test_date_money(self):
        ents = get_ner().recognize("2024年10月1日花了99.9元")
        texts = [e["text"] for e in ents]
        self.assertTrue(any("2024" in t for t in texts))
        self.assertTrue(any("99.9" in t for t in texts))


class TestSentiment(unittest.TestCase):
    def test_positive(self):
        r = get_sentiment().analyze("这个产品非常好用，我很喜欢")
        self.assertEqual(r["polarity"], "positive")

    def test_negative(self):
        r = get_sentiment().analyze("服务态度很差，令人失望")
        self.assertEqual(r["polarity"], "negative")


class TestSummarizer(unittest.TestCase):
    def test_shorter(self):
        text = ("自然语言处理是人工智能的重要分支。它研究如何让计算机理解语言。"
                "分词是基础任务。词性标注是另一个任务。")
        r = get_summarizer().summarize(text, ratio=0.5)
        self.assertTrue(len(r["summary"]) < len(text))
        self.assertTrue(r["top_indices"])


class TestTranslator(unittest.TestCase):
    def test_zh2en(self):
        r = get_translator().translate("我喜欢机器学习", "zh2en")
        self.assertIn("machine learning", r["translation"].lower())

    def test_en2zh(self):
        r = get_translator().translate("I like China", "en2zh")
        self.assertTrue(r["translation"])


class TestKeywords(unittest.TestCase):
    def test_extract(self):
        r = get_keywords().extract("自然语言处理是人工智能的重要分支", top_k=5)
        self.assertTrue(r["keywords"])
        for k in r["keywords"]:
            self.assertIn("word", k)
            self.assertIn("score", k)


class TestEmbeddings(unittest.TestCase):
    def test_train_nearest(self):
        texts = [
            "自然语言处理是人工智能的重要分支",
            "机器学习是人工智能的核心技术",
            "深度学习推动了人工智能的发展",
            "分词是自然语言处理的基础任务",
        ] * 3
        emb = get_embeddings()
        emb.train(texts, vocab_size=60, dim=8, window=3, min_count=1)
        self.assertTrue(emb.vocab)
        self.assertTrue(emb.vectors)
        # 近邻应返回词且不包含自身
        nb = emb.nearest(emb.vocab[0], k=3)
        self.assertTrue(nb)
        self.assertNotIn(emb.vocab[0], [n["word"] for n in nb])
        # 2D 投影
        proj = emb.project_2d()
        self.assertEqual(len(proj), len(emb.vectors))


class TestHMM(unittest.TestCase):
    def test_viterbi(self):
        hmm = HMM(["A", "B"], add_k=0.1)
        hmm.train([[(1, "A"), (2, "B")], [(1, "A"), (2, "B")], [(2, "B"), (1, "A")]])
        path = hmm.viterbi([1, 2])
        self.assertEqual(len(path), 2)
        self.assertIn(path[0], ("A", "B"))


class TestStorage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_shard_insert_query(self):
        store = ShardedStore(self.tmp, "t", shard_size=10)
        store.insert_many([{"v": i} for i in range(25)])
        self.assertEqual(store.stats()["total"], 25)
        self.assertEqual(store.stats()["shard_count"], 3)
        self.assertEqual(len(store.query(where=[("v", "gt", 20)])), 4)
        self.assertEqual(len(store.query(where=[("v", "in", [1, 2, 3])])), 3)

    def test_delete_compact(self):
        store = ShardedStore(self.tmp, "t", shard_size=10)
        ids = store.insert_many([{"v": i} for i in range(15)])
        store.delete(ids[0])
        stats = store.compact()
        self.assertEqual(stats["records"], 14)

    def test_update(self):
        store = ShardedStore(self.tmp, "t", shard_size=10)
        ids = store.insert_many([{"v": i, "name": f"n{i}"} for i in range(12)])
        ok = store.update(ids[5], {"v": 500})
        self.assertTrue(ok)
        rec = store.get(ids[5])
        self.assertEqual(rec["v"], 500)
        self.assertEqual(rec["name"], "n5")   # 未提及字段保留
        self.assertEqual(rec["id"], ids[5])
        self.assertEqual(store.stats()["total"], 12)  # 总数不变
        # 更新不存在的记录 / 已删除记录返回 False
        self.assertFalse(store.update("no_such_id", {"v": 1}))
        store.delete(ids[6])
        self.assertFalse(store.update(ids[6], {"v": 1}))

    def test_concurrent_insert(self):
        store = ShardedStore(self.tmp, "t", shard_size=20)
        errors = []

        def worker(offset):
            try:
                store.insert_many([{"v": offset * 1000 + i} for i in range(30)])
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertFalse(errors)
        self.assertEqual(store.stats()["total"], 180)

    def test_registry_tasks(self):
        reg = StoreRegistry(self.tmp)
        reg.task("a").insert({"x": 1})
        reg.task("b").insert({"x": 2})
        # 造一个非存储目录，不应被识别为任务
        os.makedirs(os.path.join(self.tmp, "models"))
        self.assertEqual(reg.tasks(), ["a", "b"])


class TestClustering(unittest.TestCase):
    """话题聚类：确定性、自动 k、增量更新、标签与内容一致。"""

    # 三个话题、簇大小不均（5/3/2），另加一篇无有效词文档
    DOCS = {
        "t1": "人工智能模型在图像识别任务上取得突破，算法准确率再创新高",
        "t2": "机器学习算法驱动的推荐系统上线，人工智能平台用户增长明显",
        "t3": "深度学习框架发布新版本，模型训练效率大幅提升",
        "t4": "人工智能技术加速落地，机器学习模型部署成本持续下降",
        "t5": "神经网络模型结构不断改进，图像识别算法效果显著",
        "s1": "足球联赛焦点战，主队前锋梅开二度，球队赢得比赛",
        "s2": "篮球联赛季后赛，后卫命中关键三分，球队赢得比赛晋级",
        "s3": "足球杯赛爆出冷门，小球队淘汰卫冕冠军，前锋打进制胜进球",
        "f1": "红烧肉讲究火候，五花肉慢炖入味，是经典的家常味道",
        "f2": "清蒸鱼讲究食材新鲜，火候恰到好处，味道鲜美可口",
        "x1": "的了吗在是和有",
    }

    def setUp(self):
        self.clusterer = DocumentClusterer()
        self.terms = {d: self.clusterer.tokenize(t) for d, t in self.DOCS.items()}

    def test_deterministic(self):
        """同一批文档多次聚类，结果必须完全一致。"""
        r1 = self.clusterer.cluster(self.terms)
        r2 = self.clusterer.cluster(self.terms)
        self.assertEqual(r1["assignments"], r2["assignments"])
        self.assertEqual([c["label"] for c in r1["clusters"]],
                         [c["label"] for c in r2["clusters"]])
        self.assertEqual(r1["distances"], r2["distances"])

    def test_auto_k_recovers_topics(self):
        """自动估计话题数，应大致还原三个真实话题。"""
        r = self.clusterer.cluster(self.terms)
        self.assertTrue(r["auto_k"])
        self.assertGreaterEqual(r["n_clusters"], 2)
        self.assertLessEqual(r["n_clusters"], 6)
        # 同类文档应同簇：t1..t5 同簇，s1..s3 同簇，f1/f2 同簇
        a = r["assignments"]
        self.assertEqual(len({a[d]["cluster"] for d in ("t1", "t2", "t3", "t4", "t5")}), 1)
        self.assertEqual(len({a[d]["cluster"] for d in ("s1", "s2", "s3")}), 1)
        self.assertEqual(a["f1"]["cluster"], a["f2"]["cluster"])
        # 不同类不同簇
        self.assertNotEqual(a["t1"]["cluster"], a["s1"]["cluster"])
        self.assertNotEqual(a["t1"]["cluster"], a["f1"]["cluster"])
        # 无有效词的文档被排除
        self.assertIn("x1", r["excluded"])

    def test_user_specified_k(self):
        r = self.clusterer.cluster(self.terms, n_clusters=3)
        self.assertEqual(r["n_clusters"], 3)
        self.assertFalse(r["auto_k"])
        # 指定 k 时同样确定
        r2 = self.clusterer.cluster(self.terms, n_clusters=3)
        self.assertEqual(r["assignments"], r2["assignments"])

    def test_keywords_match_content(self):
        """每簇关键词必须真实出现在簇内文档中。"""
        r = self.clusterer.cluster(self.terms, n_clusters=3)
        for c in r["clusters"]:
            members = [d for d, a in r["assignments"].items()
                       if a["cluster"] == c["id"]]
            member_text = "".join(self.DOCS[d] for d in members)
            self.assertTrue(c["keywords"])
            for kw in c["keywords"][:3]:
                self.assertIn(kw["word"], member_text)
            # 代表文档属于本簇
            for rep in c["representatives"]:
                self.assertEqual(r["assignments"][rep["doc_id"]]["cluster"], c["id"])

    def test_cluster_distances(self):
        r = self.clusterer.cluster(self.terms, n_clusters=3)
        dist = r["distances"]
        self.assertEqual(len(dist), 3)
        for i in range(3):
            self.assertAlmostEqual(dist[i][i], 0.0)
            for j in range(3):
                self.assertAlmostEqual(dist[i][j], dist[j][i])
        # 地图坐标数量与簇数一致
        self.assertTrue(all(len(c["center"]) == 2 for c in r["clusters"]))

    def test_incremental_update(self):
        """增删改只重算受影响文档；语料未变时复用结果。"""
        idx = ClusterIndex(clusterer=DocumentClusterer())
        out1 = idx.update(self.DOCS)
        self.assertFalse(out1["reused"])
        self.assertEqual(out1["updates"]["added"], len(self.DOCS))
        self.assertEqual(out1["updates"]["vectorized"], len(self.DOCS))

        # 语料未变：直接复用，不重算
        out2 = idx.update(self.DOCS)
        self.assertTrue(out2["reused"])
        self.assertEqual(out2["updates"]["vectorized"], 0)
        self.assertEqual(out1["result"]["assignments"],
                         out2["result"]["assignments"])

        # 修改一篇 + 新增一篇：只有这两篇重新分词
        docs2 = dict(self.DOCS)
        docs2["t1"] = "人工智能与机器学习推动科技发展，大模型参数持续增长"
        docs2["n1"] = "股市三大指数收涨，科技板块领涨，成交额放大"
        out3 = idx.update(docs2)
        self.assertFalse(out3["reused"])
        self.assertEqual(out3["updates"]["changed"], 1)
        self.assertEqual(out3["updates"]["added"], 1)
        self.assertEqual(out3["updates"]["vectorized"], 2)

        # 删除一篇
        docs3 = dict(docs2)
        del docs3["x1"]
        out4 = idx.update(docs3)
        self.assertEqual(out4["updates"]["removed"], 1)
        self.assertEqual(out4["updates"]["vectorized"], 0)

    def test_incremental_stability_across_reload(self):
        """状态序列化 / 反序列化后，同一语料结果保持一致。"""
        idx = ClusterIndex(clusterer=DocumentClusterer())
        out1 = idx.update(self.DOCS)
        idx2 = ClusterIndex(clusterer=DocumentClusterer(), state=idx.state)
        out2 = idx2.update(self.DOCS)
        self.assertTrue(out2["reused"])
        self.assertEqual(out1["result"]["fingerprint"],
                         out2["result"]["fingerprint"])

    def test_single_and_empty(self):
        one = self.clusterer.cluster({"a": self.clusterer.tokenize("人工智能")})
        self.assertEqual(one["n_clusters"], 1)
        with self.assertRaises(ValueError):
            self.clusterer.cluster({"a": {}, "b": {}})


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.engine = PipelineEngine().register_builtin()

    def test_run_chain(self):
        cfg = {"name": "p", "stages": [
            {"name": "segment"}, {"name": "pos"}, {"name": "sentiment"}]}
        out = self.engine.build(cfg).run({"text": "这个产品非常好用"})
        self.assertIn("words", out)
        self.assertIn("pos", out)
        self.assertIn("sentiment", out)

    def test_batch(self):
        cfg = {"name": "p", "stages": [{"name": "segment"}, {"name": "keywords"}]}
        results = self.engine.run_batch(
            cfg, ["今天天气很好", "这个产品非常好用"], max_workers=2)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["ok"] for r in results))

    def test_cycle_detected(self):
        cfg = {"name": "p", "stages": [
            {"name": "segment", "deps": ["pos"]},
            {"name": "pos", "deps": ["segment"]},
        ]}
        with self.assertRaises(PipelineError):
            self.engine.build(cfg)

    def test_missing_stage(self):
        cfg = {"name": "p", "stages": [{"name": "not_exist"}]}
        with self.assertRaises(PipelineError):
            self.engine.build(cfg)


if __name__ == "__main__":
    unittest.main()
