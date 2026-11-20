# -*- coding: utf-8 -*-
"""
《西虹市首富》猫眼电影评论 —— 数据清洗 / 脱敏 / 中文预处理 / 双模型情感分析
课程：文本挖掘技术与应用 期末作品
输出：
  maoyan_cleaned_senti.csv   清洗脱敏 + 分词 + 双模型情感结果
  stats.json                 报告所需的全部统计口径
  figures/*.png              可视化图表
"""
import os, re, json, hashlib, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import jieba
from snownlp import SnowNLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             f1_score, classification_report, confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
# 读取脱敏清洗后的数据（已上传 GitHub 的文件）
SRC = os.path.join(BASE, "data", "maoyan_cleaned_senti.csv") 
FIG = os.path.join(BASE, "figures") # 图表输出目录
os.makedirs(FIG, exist_ok=True)

# ---------- 中文字体 ----------
_candidates = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "MiSans",
               "Droid Sans Fallback", "SimHei", "DejaVu Sans"]
_available = set(f.name for f in font_manager.fontManager.ttflist)
_font = next((f for f in _candidates if f in _available), "DejaVu Sans")
plt.rcParams["font.sans-serif"] = [_font, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.bbox"] = "tight"
print("[font]", _font)

# ---------- 1. 载入 ----------
df = pd.read_csv(SRC, encoding="utf-8-sig")
df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
df["_rid"] = range(len(df))
N0 = len(df)
log = []                      # 清洗流水：(步骤, 剔除数, 剩余数, 说明)
def mark(step, before, after, note):
    log.append(dict(step=step, removed=int(before - after), remain=int(after), note=note))

# ---------- 2. 结构校验 / 缺失值与异常值 ----------
b = len(df)
df["评分"] = pd.to_numeric(df["评分"], errors="coerce")
df["评论时间"] = pd.to_datetime(df["评论时间"], errors="coerce")
df["评论内容"] = df["评论内容"].fillna("").astype(str)
df["用户昵称"] = df["用户昵称"].fillna("").astype(str)
df["用户城市"] = df["用户城市"].fillna("").astype(str).replace("", "未知")
# 昵称缺失填占位；城市缺失不删样本，统一记为「未知」
n_nick_blank = (df["用户昵称"].str.strip() == "").sum()
n_city_blank = (df["用户城市"] == "未知").sum()
df.loc[df["用户昵称"].str.strip() == "", "用户昵称"] = "匿名用户"

bad = (
    df["评分"].isna()                       # 评分缺失
    | (df["评分"] < 0.5) | (df["评分"] > 5.0)   # 超出 5 分制合法区间
    | df["评论时间"].isna()                 # 时间无法解析
    | (df["评论内容"].str.strip() == "")     # 评论正文空
    | (df["评论内容"].str.strip().str.len() < 2)  # 正文过短
)
n_bad_score = int((df["评分"].isna() | (df["评分"] < 0.5) | (df["评分"] > 5.0)).sum())
n_bad_time = int(df["评论时间"].isna().sum())
n_bad_text = int(((df["评论内容"].str.strip() == "") | (df["评论内容"].str.strip().str.len() < 2)).sum())
df = df[~bad].copy()
mark("结构校验：评分/时间/正文缺失与异常值剔除", b, len(df),
     f"评分异常 {n_bad_score} 条、时间异常 {n_bad_time} 条、正文空或过短 {n_bad_text} 条；昵称空 {n_nick_blank} 条填「匿名用户」，城市空 {n_city_blank} 条填「未知」")

# ---------- 3. 去重 ----------
b = len(df)
df = df.sort_values("评论时间").drop_duplicates(subset=["评论内容"], keep="first").copy()
n_dup = b - len(df)
mark("去重：按评论内容保留最早一条", b, len(df), f"重复评论 {n_dup} 条（完全重复行 0 条）")

# ---------- 4. 正则化 / 乱码修复 ----------
EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
    "\U0000FE0F\U0000200D\U00002190-\U000021FF\U00002B00-\U00002BFF\u200b-\u200f]+",
    flags=re.UNICODE)
MOJI = re.compile(r"[\u00c0-\u00ff\u0080-\u00bf]+")

def fix_mojibake(s):
    """修复 UTF-8 被误按 Latin-1 解码产生的 emoji 乱码（如 ð -> 😊）"""
    def _d(m):
        try:
            return m.group(0).encode("latin-1").decode("utf-8")
        except Exception:
            return ""
    return MOJI.sub(_d, s)

def normalize(s):
    s = fix_mojibake(s)
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = re.sub(r"<[^>]+>", " ", s)                       # HTML 标签
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)          # URL
    s = re.sub(r"\[[^\]]{1,20}\]", " ", s)                # [对象] 之类模板占位
    s = EMOJI.sub(" ", s)                                 # 表情符号
    # 全角 -> 半角
    s = "".join(chr(ord(ch) - 0xFEE0) if 0xFF01 <= ord(ch) <= 0xFF5E else ch for ch in s)
    s = re.sub(r"[^一-龥a-zA-Z0-9，。！？、；：,.!?;:～~“”‘’（）()《》\- ]", " ", s)
    s = re.sub(r"([，。！？、；：,.!?;:）)])\1{2,}", r"\1", s)   # 连续标点
    s = re.sub(r"\s+", " ", s).strip()
    return s

df["清洗后文本"] = df["评论内容"].map(normalize)

# ---------- 5. 去噪：跨品类串评 + 领域无关噪声 ----------
# 5.1 显式跨品类黑名单（餐饮 / 外卖 / 数码 / 美妆 / 酒店等串评）
BLACK = ["食材", "外卖", "配送", "手机壳", "面膜", "份量", "上菜", "调味", "火候",
         "味道超好", "干净卫生", "这家店", "餐厅", "菜品", "餐具", "热乎配送",
         "汤只有", "vivo", "电池续航", "拍照效果", "包装精美", "价格便宜", "服务态度",
         "及时送达", "交通也方便", "漱口杯", "光看就馋人", "招牌菜", "口感层次",
         "色泽诱人", "一进门", "暖色调", "性价比很高", "到店", "这个价格",
         "真的很划算", "每年都会来", "朋友相聚", "聚会必备", "适合来这",
         "激发起我出门", "屏幕色彩", "按键的设计", "一触即碎", "冒热气",
         "细节处理得很到位", "主动帮我换了", "汤只有余温", "菜也凉", "心都凉",
         "改善伙食", "漏汤", "没吃几口", "要么咸要么淡", "食堂", "老顾客",
         "一如既往的好吃", "味道超正", "菜品很全", "吃的安心放心"]

# 人工复核确认的乱码 / 无关串评片段（不针对影片内容）
NOISE_FRAG = ["这两种方法可以", "球球球球", "海鲶鱼", "本子龙", "通知书啊",
              "酒水接电话", "吗丁啉", "橄榄 继续加油", "太累了热土", "抓紧时间吧大海蛇",
              "我也不知道我为什么要", "这是一条评论", "一次一片的看我",
              "你的技术看到没", "八九块就咯啦", "没有本子龙安装"]

# 影视领域 / 通用评价保留词：短文本需至少命中一个，长文本直接保留（避免过度过滤）
KEEP = ["好看", "不错", "还行", "可以", "挺好", "很棒", "太好", "好得很", "棒", "推荐",
        "值得", "喜欢", "满意", "好评", "经典", "点赞", "精彩", "完美", "震撼", "惊艳",
        "搞笑", "笑", "喜剧", "电影", "影片", "剧情", "演员", "演技", "画面", "镜头",
        "配乐", "节奏", "包袱", "梗", "二刷", "重刷", "三刷", "观影", "影院", "台词",
        "角色", "表演", "导演", "剧本", "故事", "结局", "反转", "桥段", "名场面",
        "沈腾", "王多鱼", "西虹市", "首富", "开心麻花", "含腾量", "尿点", "爆米花",
        "夏洛", "力宏", "国产", "作品", "题材", "看完", "看了", "去看", "看过",
        "爽剧", "爽", "看爽", "合家欢", "有趣", "有意思", "欢乐", "感动", "深意",
        "寓意", "反思", "讽刺", "解压", "治愈", "上头", "绝", "赞", "过瘾", "失望",
        "难看", "垃圾", "无聊", "一般", "尬", "烂", "差", "爱看", "幽默", "乐", "哭",
        "泪", "质感", "特效", "视听", "审美", "风格", "共鸣", "投入", "代入", "花钱",
        "十亿", "金钱", "财富", "底线", "善良", "本心", "真情", "人性", "荒诞", "幻想",
        "三观", "低俗", "拖沓", "夸张", "翻拍", "原定女主", "半颗灰星", "印象",
        "评分", "评价", "值得一看", "真快", "快了", "值", "推", "牛",
        "腾哥", "好样", "好样得", "麻花", "含腾"]

def is_noise(t):
    if len(t) < 2:
        return True
    if any(k in t for k in BLACK):
        return True
    if any(k in t for k in NOISE_FRAG):
        return True
    # 短文本若无任何影视 / 评价语义词，判为无关噪声；长文本保留，避免误删口语化短评
    if len(t) < 14 and not any(k in t for k in KEEP):
        return True
    return False

b = len(df)
df["_noise"] = df["清洗后文本"].map(is_noise)
noise_examples = df.loc[df["_noise"], "清洗后文本"].head(12).tolist()
df = df[~df["_noise"]].copy()
mark("去噪：跨品类串评、乱码与领域无关文本过滤", b, len(df),
     f"剔除 {b - len(df)} 条；规则=跨品类词表（餐饮/数码/酒店等）+ 人工复核噪声片段表 + 短文本领域相关性校验（长文本一律保留，避免误删口语化短评）")

# ---------- 6. 数据脱敏 ----------
def pseudonym(x):
    return "U" + hashlib.sha256(("maoyan|" + str(x)).encode("utf-8")).hexdigest()[:10].upper()

PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
EMAIL = re.compile(r"[\w.\-]+@[\w\-]+\.[a-zA-Z]{2,}")
URL   = re.compile(r"https?://\S+|www\.\S+")
ATUSER = re.compile(r"@[\w\u4e00-\u9fa5\-]{2,20}")
WX    = re.compile(r"(?:微信|weixin|vx|V信)\s*[:：]?\s*[\w\-]{5,20}")

def desensitize(s):
    s = PHONE.sub("[手机号]", s)
    s = EMAIL.sub("[邮箱]", s)
    s = URL.sub("[链接]", s)
    s = ATUSER.sub("[用户]", s)
    s = WX.sub("[联系方式]", s)
    return s

df["用户ID"] = df["用户昵称"].map(pseudonym)
df["评论原文_脱敏"] = df["评论内容"].map(desensitize)
df["清洗后文本"] = df["清洗后文本"].map(desensitize)
n_phone = int(df["评论原文_脱敏"].str.contains("\[手机号\]").sum())
n_other = int(df["评论原文_脱敏"].str.contains("\[(?:邮箱|链接|用户|联系方式)\]").sum())
mark("数据脱敏：昵称哈希化 + 正文联系方式掩码", len(df), len(df),
     f"昵称全部替换为不可逆 SHA-256 短 ID；正文命中手机号 {n_phone} 处、其它标识 {n_other} 处；不导出映射表")

# ---------- 7. 中文文本预处理：分词 + 停用词 ----------
for w in ["西虹市首富", "王多鱼", "沈腾", "开心麻花", "夏洛特烦恼", "含腾量",
          "无尿点", "笑点", "二刷", "重刷", "三刷", "名场面", "爆米花", "合家欢",
          "一个月花光十亿", "一个月花光", "花钱", "暴富"]:
    jieba.add_word(w)

# 停用词只去功能词、标点与无区分度的通用名词；
# 情感词（好看 / 不错 / 推荐 / 搞笑 / 垃圾 等）必须保留，否则短评会退化为零特征。
STOP = set("""的 了 是 在 我 有 和 就 不 人 都 一 上 也 很 到 说 要 去 你 会 着 没有 看 好
自己 这 那 之 与 及 或 但 而 且 把 被 给 让 从 对 为 以 其 中 里 外 时 后 前 下 多 少 大 小
啊 呀 吧 呢 吗 哦 嗯 哈 啦 嘛 哇 唉 哎 噢 咯 喔 呵 呗 哒 哟 嘞 嘀
这个 那个 这些 那些 这么 那么 什么 怎么 这样 那样 如此 一些 一点 有点
感觉 觉得 还是 还有 真的 确实 完全 整体 基本 比较 相当 十分 格外 特别 非常 超级
电影 影片 片子 部 电影里 观影 的时候 时候 一部 这部 那部 里面 之后 之前
一个 一次 一直 一样 一下 一直 已经 就是 只是 可是 不过 但是 因为 所以 如果 虽然
大家 我们 你们 他们 它 他 她 谁 哪 哪个 哪里 几 多少 更 最 太 又 再 还 才
嗯嗯 哈哈 哈哈哈 呵呵 嘻嘻 呜呜 嘿嘿 么么哒 棒棒哒
， 。 ！ ？ 、 ； ： , . ! ? ; : （ ） ( ) 《 》 “ ” ‘ ’ … ～ ~ - — 的 地 得""".split())
STOP |= set("abcdefghijklmnopqrstuvwxyz0123456789")

SINGLE_KEEP = {"笑", "尬", "烂", "差", "爽", "燃", "绝", "赞", "梗"}
def cut(s):
    out = []
    for w in jieba.lcut(s):
        if not w.strip() or w in STOP or re.fullmatch(r"[\W\d_]+", w):
            continue
        if len(w) > 1 or w in SINGLE_KEEP:
            out.append(w)
    return out

df["分词结果"] = df["清洗后文本"].map(lambda s: " ".join(cut(s)))
df["词数"] = df["分词结果"].map(lambda s: len(s.split()))
mark("中文预处理：正则化、jieba 分词、去停用词", len(df), len(df),
     "自定义词典补充片名/角色/喜剧术语；停用词表含标点、代词、程度副词与通用评价词")

df = df.reset_index(drop=True)   # 保证后续 numpy 位置索引与 DataFrame 标签一致

# ---------- 8. 评分弱标签（模型参照基准） ----------
def lab(r):
    if r >= 4.0: return "正面"
    if r >= 3.0: return "中性"
    return "负面"
df["评分情感标签"] = df["评分"].map(lab)

# ---------- 9. 方法一：SnowNLP 情感打分（无监督预训练模型） ----------
df["SnowNLP情感分值"] = df["清洗后文本"].map(lambda s: round(float(SnowNLP(s).sentiments), 4))
TH_POS, TH_NEG = 0.60, 0.40
def snow_polar(v):
    return "正面" if v >= TH_POS else ("负面" if v <= TH_NEG else "中性")
df["SnowNLP极性"] = df["SnowNLP情感分值"].map(snow_polar)

# ---------- 10. 方法二：TF-IDF + 朴素贝叶斯（有监督，弱标签来自评分） ----------
CLS = ["负面", "中性", "正面"]
X_text = df["分词结果"].tolist()
y = df["评分情感标签"].tolist()
idx_train, idx_test = train_test_split(
    np.arange(len(df)), test_size=0.30, random_state=42, stratify=y)
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95,
                      sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
X = vec.fit_transform(X_text)
Xtr, Xte = X[idx_train], X[idx_test]
ytr = np.array(y)[idx_train]; yte = np.array(y)[idx_test]

# 三分类样本极不均衡（正面约 94%），默认先验会把所有样本推向正面。
# 主模型采用均衡类别先验（fit_prior=False + 均匀 class_prior），另保留默认先验作为消融基线。
nb_base = MultinomialNB(alpha=0.5).fit(Xtr, ytr)
pred_base_te = nb_base.predict(Xte)
nb = MultinomialNB(alpha=0.5, fit_prior=False,
                   class_prior=np.array([1 / 3, 1 / 3, 1 / 3]))
nb.fit(Xtr, ytr)
pred_te = nb.predict(Xte)
pred_all = nb.predict(X)
df["NB极性"] = pred_all
df["_split"] = "训练集"
df.loc[idx_test, "_split"] = "测试集"
df["NB测试预测"] = ""
df.loc[idx_test, "NB测试预测"] = pred_te

def metrics(yt, yp):
    p, r, f, _ = precision_recall_fscore_support(yt, yp, labels=CLS, zero_division=0)
    return dict(
        accuracy=round(float(accuracy_score(yt, yp)), 4),
        macro_p=round(float(p.mean()), 4), macro_r=round(float(r.mean()), 4),
        macro_f1=round(float(f.mean()), 4),
        weighted_f1=round(float(f1_score(yt, yp, average="weighted", zero_division=0)), 4),
        per_class={c: dict(p=round(float(p[i]), 4), r=round(float(r[i]), 4),
                           f1=round(float(f[i]), 4)) for i, c in enumerate(CLS)}
    )

m_nb = metrics(yte, pred_te)
m_nb_base = metrics(yte, pred_base_te)
m_snow_test = metrics(yte, df.iloc[idx_test]["SnowNLP极性"].tolist())
m_snow_all = metrics(y, df["SnowNLP极性"].tolist())
m_nb_all = metrics(y, pred_all)
cm = confusion_matrix(yte, pred_te, labels=CLS)

# 二分类辅助：正面 vs 非正面（中/负合并），检验少数类识别能力
ybin = np.array(["正面" if v == "正面" else "非正面" for v in y])
itr2, ite2 = train_test_split(np.arange(len(df)), test_size=0.30,
                              random_state=42, stratify=ybin)
nb2 = MultinomialNB(alpha=0.5).fit(X[itr2], ybin[itr2])
pb2 = nb2.predict(X[ite2])
m_bin = dict(
    accuracy=round(float(accuracy_score(ybin[ite2], pb2)), 4),
    f1_nonpos=round(float(f1_score(ybin[ite2], pb2, pos_label="非正面", zero_division=0)), 4),
    recall_nonpos=round(float(precision_recall_fscore_support(
        ybin[ite2], pb2, labels=["非正面"], zero_division=0)[1][0]), 4)
)

# ---------- 10.1 评分弱标签与文本情感的一致性校验 ----------
conflict_low = int(((df["评分"] < 3.0) & (df["SnowNLP情感分值"] >= TH_POS)).sum())
conflict_high = int(((df["评分"] >= 4.0) & (df["SnowNLP情感分值"] <= TH_NEG)).sum())
conflict_total = conflict_low + conflict_high
conflict_examples = (df[((df["评分"] < 3.0) & (df["SnowNLP情感分值"] >= TH_POS))
                        | ((df["评分"] >= 4.0) & (df["SnowNLP情感分值"] <= TH_NEG))]
                     .head(5)[["评分", "清洗后文本", "SnowNLP情感分值"]]
                     .values.tolist())

# ---------- 11. 主题词统计 ----------
def cnt_word(kw):
    return int(df["清洗后文本"].str.contains(kw, regex=False).sum())
THEME = ["沈腾", "笑点", "搞笑", "喜剧", "剧情", "二刷", "重刷", "三刷", "王多鱼",
         "十亿", "金钱", "财富", "底线", "善良", "本心", "讽刺", "演技", "画面",
         "节奏", "包袱", "合家欢", "解压", "翻拍", "拖沓", "尬", "低俗", "夸张",
         "无聊", "配乐", "荒诞", "幻想", "爽剧", "反转", "经典", "国产"]
theme_cnt = {k: cnt_word(k) for k in THEME}

def top_words(mask, n=15):
    from collections import Counter
    c = Counter()
    for s in df.loc[mask, "分词结果"]:
        c.update(s.split())
    return c.most_common(n)
mask_pos = df["评分情感标签"] == "正面"
mask_neg = df["评分情感标签"].isin(["负面", "中性"])
top_pos = top_words(mask_pos)
top_neg = top_words(mask_neg)

# 典型评论（脱敏后，供报告引述）
_pos_cand = df[mask_pos & (df["清洗后文本"].str.len() > 40)].copy()
_pos_cand["_len"] = _pos_cand["清洗后文本"].str.len()
pos_quotes = _pos_cand.sort_values("_len", ascending=False).head(3)["清洗后文本"].tolist()
neg_quotes = (df[mask_neg].sort_values("评分")
              .head(6)["清洗后文本"].tolist())

# ---------- 12. 分布统计 ----------
N = len(df)
def dist(series):
    vc = series.value_counts()
    return {k: dict(n=int(vc.get(k, 0)), pct=round(float(vc.get(k, 0) / N * 100), 2))
            for k in CLS}
d_label = dist(df["评分情感标签"])
d_snow = dist(df["SnowNLP极性"])
d_nb = dist(df["NB极性"])
rating_dist = {str(k): int(v) for k, v in df["评分"].value_counts().sort_index().items()}
monthly = (df.set_index("评论时间").resample("MS").size())
city_top = df["用户城市"].value_counts().head(10).to_dict()

stats = dict(
    n_raw=N0, n_clean=N, n_removed=N0 - N,
    clean_log=log, noise_examples=noise_examples,
    rating_dist=rating_dist, rating_mean=round(float(df["评分"].mean()), 4),
    rating_median=float(df["评分"].median()),
    snow_mean=round(float(df["SnowNLP情感分值"].mean()), 4),
    snow_median=round(float(df["SnowNLP情感分值"].median()), 4),
    snow_std=round(float(df["SnowNLP情感分值"].std()), 4),
    snow_mean_by_label={c: round(float(df.loc[df["评分情感标签"] == c, "SnowNLP情感分值"].mean()), 4)
                        for c in CLS},
    snow_th=dict(pos=TH_POS, neg=TH_NEG),
    dist_label=d_label, dist_snow=d_snow, dist_nb=d_nb,
    m_nb=m_nb, m_nb_base=m_nb_base,
    m_snow_test=m_snow_test, m_snow_all=m_snow_all, m_nb_all=m_nb_all,
    m_bin=m_bin, cm=cm.tolist(), classes=CLS,
    n_train=len(idx_train), n_test=len(idx_test),
    vocab_size=int(len(vec.vocabulary_)),
    conflict_low=conflict_low, conflict_high=conflict_high,
    conflict_total=conflict_total, conflict_examples=conflict_examples,
    theme_cnt=theme_cnt, top_pos=top_pos, top_neg=top_neg,
    pos_quotes=pos_quotes, neg_quotes=neg_quotes,
    monthly={str(k.date())[:7]: int(v) for k, v in monthly.items() if v > 0},
    city_top={k: int(v) for k, v in city_top.items()},
    n_phone=n_phone, n_other=n_other,
    date_min=str(df["评论时间"].min()), date_max=str(df["评论时间"].max()),
    avg_words=round(float(df["词数"].mean()), 2),
)
with open(os.path.join(BASE, "stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

# ---------- 13. 导出清洗结果 ----------
out_cols = ["用户ID", "评分", "评论原文_脱敏", "清洗后文本", "分词结果", "词数",
            "用户城市", "评论时间", "评分情感标签", "SnowNLP情感分值", "SnowNLP极性",
            "NB极性", "_split", "NB测试预测"]
out = df[out_cols].rename(columns={"_split": "数据集划分"}).copy().fillna("")
out["评论时间"] = df["评论时间"].dt.strftime("%Y-%m-%d %H:%M").values
out.to_csv(os.path.join(BASE, "maoyan_cleaned_senti.csv"),
           index=False, encoding="utf-8-sig")
print("[clean]", N0, "->", N, "| label", d_label)

# ---------- 14. 可视化 ----------
C_POS, C_NEU, C_NEG = "#e05c5c", "#f0b95c", "#5b9bd5"
COLORS = [C_NEG, C_NEU, C_POS]

# 图1 评分分布
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ks = sorted(rating_dist, key=float)
ax.bar(ks, [rating_dist[k] for k in ks], color="#5b9bd5", edgecolor="white")
for i, k in enumerate(ks):
    ax.text(i, rating_dist[k] + 3, str(rating_dist[k]), ha="center", fontsize=9)
ax.set_title("图 1  清洗后评论的评分分布（5 分制）", fontsize=12)
ax.set_xlabel("评分"); ax.set_ylabel("评论数")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(os.path.join(FIG, "fig01_rating_dist.png")); plt.close(fig)

# 图2 三种口径情感极性分布
fig, ax = plt.subplots(figsize=(7.6, 4.2))
x = np.arange(3); w = 0.26
series = [("评分弱标签", d_label), ("SnowNLP", d_snow), ("朴素贝叶斯", d_nb)]
for j, (name, d) in enumerate(series):
    vals = [d[c]["pct"] for c in ["负面", "中性", "正面"]]
    bars = ax.bar(x + (j - 1) * w, vals, w, label=name,
                  color=["#7f9fc4", "#e8a35c", "#d35c5c"][j], edgecolor="white")
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 1.2, f"{v:.1f}%",
                ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(["负面", "中性", "正面"])
ax.set_ylabel("占比（%）"); ax.set_ylim(0, 105)
ax.set_title("图 2  三种口径的情感极性分布对比", fontsize=12)
ax.legend(fontsize=9, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(os.path.join(FIG, "fig02_polarity_compare.png")); plt.close(fig)

# 图3 SnowNLP 情感分值分布
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.hist(df["SnowNLP情感分值"], bins=25, color="#5b9bd5", edgecolor="white")
ax.axvline(TH_NEG, color=C_NEG, ls="--", lw=1.4, label=f"负面阈值 {TH_NEG}")
ax.axvline(TH_POS, color=C_POS, ls="--", lw=1.4, label=f"正面阈值 {TH_POS}")
ax.axvline(stats["snow_mean"], color="#333", ls="-", lw=1.2, label=f"均值 {stats['snow_mean']}")
ax.set_title("图 3  SnowNLP 情感得分分布", fontsize=12)
ax.set_xlabel("情感分值（0=负面，1=正面）"); ax.set_ylabel("评论数")
ax.legend(fontsize=9, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(os.path.join(FIG, "fig03_snow_score_dist.png")); plt.close(fig)

# 图4 按评分标签分组的 SnowNLP 得分箱线
fig, ax = plt.subplots(figsize=(7.0, 4.0))
data = [df.loc[df["评分情感标签"] == c, "SnowNLP情感分值"].values for c in ["负面", "中性", "正面"]]
bp = ax.boxplot(data, patch_artist=True, widths=0.55)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["负面", "中性", "正面"])
for patch, c in zip(bp["boxes"], COLORS):
    patch.set_facecolor(c); patch.set_alpha(0.75)
ax.axhline(0.5, color="#888", ls=":", lw=1)
ax.set_title("图 4  不同评分标签下的 SnowNLP 情感得分", fontsize=12)
ax.set_ylabel("SnowNLP 情感分值")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(os.path.join(FIG, "fig04_score_by_label.png")); plt.close(fig)

# 图5 NB 混淆矩阵（测试集）
fig, ax = plt.subplots(figsize=(5.6, 4.6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(CLS); ax.set_yticklabels(CLS)
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                color="white" if cm[i][j] > cm.max() / 2 else "#333", fontsize=12)
ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别（评分弱标签）")
ax.set_title("图 5  朴素贝叶斯测试集混淆矩阵", fontsize=12)
fig.colorbar(im, ax=ax, shrink=0.8)
fig.savefig(os.path.join(FIG, "fig05_confusion_nb.png")); plt.close(fig)

# 图6 模型指标对比（同一测试集）
fig, ax = plt.subplots(figsize=(7.6, 4.2))
names = ["准确率", "宏平均 P", "宏平均 R", "宏平均 F1", "加权 F1"]
keys = ["accuracy", "macro_p", "macro_r", "macro_f1", "weighted_f1"]
v1 = [m_snow_test[k] for k in keys]; v2 = [m_nb[k] for k in keys]
x = np.arange(len(names)); w = 0.36
ax.bar(x - w / 2, v1, w, label="SnowNLP", color="#e8a35c", edgecolor="white")
ax.bar(x + w / 2, v2, w, label="朴素贝叶斯", color="#5b9bd5", edgecolor="white")
for i, (a_, b_) in enumerate(zip(v1, v2)):
    ax.text(i - w / 2, a_ + .015, f"{a_:.2f}", ha="center", fontsize=8)
    ax.text(i + w / 2, b_ + .015, f"{b_:.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
ax.set_ylim(0, 1.12); ax.set_ylabel("指标值")
ax.set_title("图 6  两种模型在同一测试集上的性能对比", fontsize=12)
ax.legend(fontsize=9, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(os.path.join(FIG, "fig06_model_metrics.png")); plt.close(fig)

# 图7 月度评论量
fig, ax = plt.subplots(figsize=(7.6, 3.8))
mk = list(stats["monthly"].keys()); mv = list(stats["monthly"].values())
ax.plot(range(len(mk)), mv, marker="o", color="#5b9bd5", lw=1.8)
ax.set_xticks(range(len(mk))); ax.set_xticklabels(mk, rotation=45, ha="right", fontsize=8)
ax.set_title("图 7  评论时间分布（按月）", fontsize=12)
ax.set_ylabel("评论数"); ax.grid(axis="y", alpha=.3)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(os.path.join(FIG, "fig07_monthly.png")); plt.close(fig)

# 图8 城市 TOP10
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ck = list(city_top.keys())[::-1]; cv = list(city_top.values())[::-1]
ax.barh(ck, cv, color="#7f9fc4", edgecolor="white")
for i, v in enumerate(cv):
    ax.text(v + .3, i, str(v), va="center", fontsize=8)
ax.set_title("图 8  评论用户城市分布 TOP10", fontsize=12)
ax.set_xlabel("评论数")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(os.path.join(FIG, "fig08_city_top10.png")); plt.close(fig)

# 图9 正面/负面高频词
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
for ax_, (title, tw, color) in zip(axes, [("正面评论高频词", top_pos[:12], C_POS),
                                          ("中性与负面评论高频词", top_neg[:12], C_NEG)]):
    ws = [w for w, c in tw][::-1]; cs = [c for w, c in tw][::-1]
    ax_.barh(ws, cs, color=color, alpha=.85, edgecolor="white")
    ax_.set_title(title, fontsize=11); ax_.set_xlabel("词频")
    ax_.tick_params(labelsize=9)
    ax_.spines[["top", "right"]].set_visible(False)
fig.suptitle("图 9  不同情感评论的主题词对比", fontsize=12)
fig.savefig(os.path.join(FIG, "fig09_topwords.png")); plt.close(fig)

print("[figs]", len(os.listdir(FIG)), "png files")
print("[metrics] NB test", m_nb["accuracy"], m_nb["macro_f1"],
      "| Snow test", m_snow_test["accuracy"], m_snow_test["macro_f1"])
print("[done]")
