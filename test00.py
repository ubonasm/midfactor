import streamlit as st
import pandas as pd
import spacy
import re
import json
import base64
from io import StringIO
import os

# アプリのタイトルとスタイル設定
st.set_page_config(page_title="授業記録分析ツール", layout="wide")
st.title("授業記録分析ツール")
st.markdown("発言内容を分析し、特定のパターンにブラケットを付けます")

# spaCyモデルのロード
@st.cache_resource
def load_nlp_model():
    try:
        return spacy.load("ja_core_news_sm")
    except OSError:
        st.info("日本語モデルをダウンロードしています...")
        spacy.cli.download("ja_core_news_sm")
        return spacy.load("ja_core_news_sm")

nlp = load_nlp_model()

# デフォルトの概念辞書
default_concept_dict = {
    # 教科・単元・授業概念
    "教科・単元・授業概念": [
        "数", "足し算", "引き算", "掛け算", "割り算", "方程式", "関数", "図形", "確率",
        "物語", "文章", "読解", "表現", "文法", "漢字", "文学", "詩",
        "実験", "観察", "生物", "化学", "物理", "地学", "元素", "反応", "エネルギー",
        "歴史", "地理", "政治", "経済", "文化", "社会", "国際", "環境",
        "単元", "授業", "学習", "教材", "カリキュラム"
    ],
    
    # 社会的概念
    "社会的概念": [
        "道徳", "正義", "権利", "責任", "平等", "自由", "尊重", "協力", "共生",
        "多様性", "持続可能性", "民主主義", "市民性", "グローバル", "地域", "伝統",
        "対話", "議論", "発表", "協力", "チームワーク", "意見", "合意形成"
    ],
    
    # 思考プロセス
    "思考プロセス": [
        "考え", "アイデア", "仮説", "予想", "推測", "分析", "評価", "判断",
        "創造", "批判的思考", "問題解決", "メタ認知", "振り返り", "計画"
    ]
}

# 概念辞書をグローバル変数として初期化
concept_dict = default_concept_dict.copy()

# 具体例を示す表現のパターン（拡張版）
example_patterns = [
    # 明示的な例示表現
    r"例えば[、,]", r"たとえば[、,]", r"具体的には", r"具体例", r"事例", r"実例", 
    # 時間や場所の具体的表現
    r"\d+月\d+日", r"昨日", r"先日", r"先週", r"今日", r"明日", r"午前", r"午後", 
    r"〜時", r"〜分", r"〜年", r"〜月", r"〜日",
    # 人物や所有物の表現
    r"〜さんは", r"〜くんは", r"〜ちゃんは", r"私は", r"僕は", r"わたしは", r"ぼくは", 
    r"私の", r"僕の", r"わたしの", r"ぼくの", r"持っている", r"買った", r"もらった",
    # 場所や店舗の表現
    r"〜店", r"〜屋", r"〜館", r"〜園", r"〜公園", r"〜学校", r"〜駅", r"〜市", r"〜県", 
    r"〜町", r"〜村", r"〜地区", r"〜センター",
    # 数量表現
    r"\d+個", r"\d+円", r"\d+人", r"\d+匹", r"\d+台", r"\d+本", r"\d+冊", r"\d+回",
    # 経験表現
    r"行った", r"見た", r"聞いた", r"感じた", r"体験", r"経験", r"やってみた", r"試した"
]

# アイデア・思い・構想を示す表現のパターン（拡張版）
idea_patterns = [
    # 思考表現
    r"思います", r"考えます", r"感じます", r"だと思う", r"ではないか", r"かもしれない", 
    r"だろう", r"でしょう", r"ようだ", r"みたいだ", r"らしい", r"っぽい",
    # 願望・希望表現
    r"したい", r"欲しい", r"希望", r"願い", r"夢", r"目標", r"理想", r"期待",
    # 提案・アイデア表現
    r"アイデア", r"提案", r"案", r"方法", r"やり方", r"工夫", r"改善", r"解決策",
    # 感情表現
    r"嬉しい", r"悲しい", r"楽しい", r"面白い", r"怖い", r"不安", r"心配", r"安心",
    # 意見表現
    r"意見", r"考え", r"見解", r"立場", r"視点", r"観点", r"主張", r"賛成", r"反対",
    # 仮定表現
    r"もし", r"仮に", r"たら", r"れば", r"なら", r"とすれば", r"と仮定すると"
]

def parse_concept_dict_file(file_content):
    """テキストファイルから概念辞書を解析する"""
    concept_dict = {}
    current_category = None
    
    for line in file_content.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):  # 空行またはコメント行をスキップ
            continue
        
        if line.startswith('[') and line.endswith(']'):
            # カテゴリ行
            current_category = line[1:-1].strip()
            concept_dict[current_category] = []
        elif current_category is not None:
            # 用語行
            terms = [term.strip() for term in line.split(',')]
            concept_dict[current_category].extend([term for term in terms if term])
    
    return concept_dict

def analyze_text_with_context(text):
    """文脈を考慮してテキストを分析し、適切なブラケットを付ける"""
    doc = nlp(text)
    
    # 文節ごとの分析結果を保存
    segments = []
    
    # 文を分割して処理
    for sent in doc.sents:
        sent_text = sent.text
        bracket_applied = False
        
        # 具体例のパターンを検出
        for pattern in example_patterns:
            if re.search(pattern, sent_text):
                sent_text = f"[{sent_text}]"
                bracket_applied = True
                break
        
        if not bracket_applied:
            # 概念辞書を使用して概念を検出
            for category, terms in concept_dict.items():
                for term in terms:
                    if term in sent_text:
                        sent_text = f"（{sent_text}）"
                        bracket_applied = True
                        break
                if bracket_applied:
                    break
        
        if not bracket_applied:
            # アイデア・思い・構想のパターンを検出
            for pattern in idea_patterns:
                if re.search(pattern, sent_text):
                    sent_text = f"〈{sent_text}〉"
                    bracket_applied = True
                    break
        
        segments.append(sent_text)
    
    # 文脈分析（隣接する文の関係を考慮）
    # 例：「私は考えました」の後に続く文は、アイデアである可能性が高い
    result = []
    idea_context = False
    
    for i, segment in enumerate(segments):
        if i > 0 and idea_context and not (segment.startswith("[") or segment.startswith("（") or segment.startswith("〈")):
            # 前の文がアイデア文脈で、現在の文にブラケットがない場合
            segment = f"〈{segment}〉"
        
        # アイデア文脈の更新
        idea_context = any(re.search(pattern, segment) for pattern in [r"考え", r"思い", r"アイデア", r"提案"])
        
        result.append(segment)
    
    return "".join(result)

def process_csv(df):
    """CSVデータを処理し、発言内容にブラケットを付ける"""
    if '発言内容' not in df.columns:
        st.error("CSVファイルに「発言内容」列が見つかりません。")
        return None
    
    # 発言内容を分析
    df['分析済み発言内容'] = df['発言内容'].apply(analyze_text_with_context)
    
    return df

def get_csv_download_link(df, filename="analyzed_data.csv"):
    """データフレームをCSVとしてダウンロードするためのリンクを生成"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">分析結果をダウンロード</a>'
    return href

def get_dict_download_link(dict_data, filename="concept_dictionary.txt"):
    """概念辞書をテキストファイルとしてダウンロードするためのリンクを生成"""
    content = ""
    for category, terms in dict_data.items():
        content += f"[{category}]\n"
        # 10個ごとに改行して見やすくする
        for i in range(0, len(terms), 10):
            content += ", ".join(terms[i:i+10]) + "\n"
        content += "\n"
    
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">概念辞書をダウンロード</a>'
    return href

def get_pattern_download_link(filename="patterns.txt"):
    """パターン設定をテキストファイルとしてダウンロードするためのリンクを生成"""
    content = "# 具体例パターン\n"
    for pattern in example_patterns:
        content += pattern + "\n"
    
    content += "\n# アイデア・思い・構想パターン\n"
    for pattern in idea_patterns:
        content += pattern + "\n"
    
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">パターン設定をダウンロード</a>'
    return href

# サイドバーの設定
st.sidebar.header("設定")

# 概念辞書のアップロード
st.sidebar.subheader("概念辞書のアップロード")
dict_file = st.sidebar.file_uploader("概念辞書ファイルをアップロード", type=["txt"])

if dict_file is not None:
    try:
        dict_content = dict_file.getvalue().decode("utf-8")
        uploaded_dict = parse_concept_dict_file(dict_content)
        if uploaded_dict:
            concept_dict = uploaded_dict
            st.sidebar.success("概念辞書を読み込みました。")
        else:
            st.sidebar.warning("有効な概念辞書が見つかりませんでした。デフォルト辞書を使用します。")
    except Exception as e:
        st.sidebar.error(f"辞書の読み込み中にエラーが発生しました: {e}")
        st.sidebar.info("デフォルト辞書を使用します。")

# 概念辞書のフォーマット例を表示
st.sidebar.subheader("概念辞書のフォーマット例")
dict_format_example = """
# 概念辞書フォーマット例
[教科・単元・授業概念]
数学, 算数, 国語, 理科, 社会
方程式, 関数, 図形, 確率, 統計

[社会的概念]
民主主義, 人権, 環境, 持続可能性
多様性, 公正, 平等, 自由, 責任

# コメント行は#で始めます
"""
st.sidebar.code(dict_format_example, language="text")

# 現在の概念辞書をダウンロード
if st.sidebar.button("現在の概念辞書をダウンロード"):
    st.sidebar.markdown(get_dict_download_link(concept_dict), unsafe_allow_html=True)

# パターン設定のアップロード
st.sidebar.subheader("パターン設定のアップロード")
pattern_file = st.sidebar.file_uploader("パターン設定ファイルをアップロード", type=["txt"])

if pattern_file is not None:
    try:
        pattern_content = pattern_file.getvalue().decode("utf-8")
        
        # パターン設定を解析
        new_example_patterns = []
        new_idea_patterns = []
        current_section = None
        
        for line in pattern_content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                if "具体例" in line:
                    current_section = "example"
                elif "アイデア" in line or "思い" in line or "構想" in line:
                    current_section = "idea"
                continue
            
            if current_section == "example":
                new_example_patterns.append(line)
            elif current_section == "idea":
                new_idea_patterns.append(line)
        
        if new_example_patterns:
            example_patterns = new_example_patterns
        if new_idea_patterns:
            idea_patterns = new_idea_patterns
            
        st.sidebar.success("パターン設定を読み込みました。")
    except Exception as e:
        st.sidebar.error(f"パターン設定の読み込み中にエラーが発生しました: {e}")

# 現在のパターン設定をダウンロード
if st.sidebar.button("現在のパターン設定をダウンロード"):
    st.sidebar.markdown(get_pattern_download_link(), unsafe_allow_html=True)

# サンプルデータの表示
st.sidebar.subheader("サンプルデータ形式")
sample_data = pd.DataFrame({
    '発言番号': [1, 2, 3, 4, 5],
    '発言者': ['教師', '生徒A', '生徒B', '生徒C', '教師'],
    '発言内容': [
        '今日は三角形の面積について学びましょう。',
        '例えば、この図形の面積はどうやって求めますか？',
        '底辺×高さ÷2だと思います。',
        '昨日、お父さんと一緒に公園で三角形の看板を見ました。',
        'みなさんの考えを聞かせてください。'
    ]
})
st.sidebar.dataframe(sample_data)

# ファイルアップロード
uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])

# サンプルデータを使用するオプション
use_sample = st.checkbox("サンプルデータを使用")

# 分析設定
st.sidebar.subheader("分析設定")
show_patterns = st.sidebar.checkbox("パターン検出設定を表示")

if show_patterns:
    st.sidebar.subheader("具体例パターン")
    example_patterns_text = st.sidebar.text_area("具体例を示す表現パターン（1行に1つ）", 
                                               "\n".join(p for p in example_patterns))
    
    st.sidebar.subheader("アイデアパターン")
    idea_patterns_text = st.sidebar.text_area("アイデア・思い・構想を示す表現パターン（1行に1つ）", 
                                            "\n".join(p for p in idea_patterns))
    
    if st.sidebar.button("パターン設定を更新"):
        example_patterns = [p.strip() for p in example_patterns_text.split("\n") if p.strip()]
        idea_patterns = [p.strip() for p in idea_patterns_text.split("\n") if p.strip()]
        st.sidebar.success("パターン設定を更新しました。")

# 分析オプション
st.sidebar.subheader("分析オプション")
enable_context_analysis = st.sidebar.checkbox("文脈分析を有効にする", value=True)
bracket_overlap_strategy = st.sidebar.radio(
    "ブラケット重複時の戦略",
    ["優先順位（具体例 > 概念 > アイデア）", "最長一致", "重複を許可（入れ子）"]
)

# 分析結果の統計
def show_analysis_stats(df):
    if 'analyzed' not in st.session_state:
        st.session_state.analyzed = True
        
        total_utterances = len(df)
        example_count = sum(1 for text in df['分析済み発言内容'] if '[' in text)
        concept_count = sum(1 for text in df['分析済み発言内容'] if '（' in text)
        idea_count = sum(1 for text in df['分析済み発言内容'] if '〈' in text)
        
        st.subheader("分析統計")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("総発言数", total_utterances)
        with col2:
            st.metric("具体例 [　]", example_count, f"{example_count/total_utterances:.1%}")
        with col3:
            st.metric("概念 （　）", concept_count, f"{concept_count/total_utterances:.1%}")
        with col4:
            st.metric("アイデア 〈　〉", idea_count, f"{idea_count/total_utterances:.1%}")
        
        # 発言者別の統計
        if '発言者' in df.columns:
            st.subheader("発言者別の分析")
            speaker_stats = {}
            
            for speaker in df['発言者'].unique():
                speaker_df = df[df['発言者'] == speaker]
                speaker_total = len(speaker_df)
                speaker_example = sum(1 for text in speaker_df['分析済み発言内容'] if '[' in text)
                speaker_concept = sum(1 for text in speaker_df['分析済み発言内容'] if '（' in text)
                speaker_idea = sum(1 for text in speaker_df['分析済み発言内容'] if '〈' in text)
                
                speaker_stats[speaker] = {
                    "総発言数": speaker_total,
                    "具体例": speaker_example,
                    "概念": speaker_concept,
                    "アイデア": speaker_idea
                }
            
            speaker_df = pd.DataFrame(speaker_stats).T
            st.dataframe(speaker_df)
            
            # 発言者別のグラフ
            st.subheader("発言者別のブラケット分布")
            speaker_chart_data = pd.DataFrame({
                "発言者": list(speaker_stats.keys()) * 3,
                "ブラケット種類": ["具体例"] * len(speaker_stats) + ["概念"] * len(speaker_stats) + ["アイデア"] * len(speaker_stats),
                "発言数": [stats["具体例"] for stats in speaker_stats.values()] + 
                         [stats["概念"] for stats in speaker_stats.values()] + 
                         [stats["アイデア"] for stats in speaker_stats.values()]
            })
            
            st.bar_chart(speaker_chart_data, x="発言者", y="発言数", color="ブラケット種類")

# メイン処理
if uploaded_file is not None:
    # アップロードされたファイルを処理
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("アップロードされたデータ")
        st.dataframe(df)
        
        analyzed_df = process_csv(df)
        if analyzed_df is not None:
            st.subheader("分析結果")
            st.dataframe(analyzed_df)
            st.markdown(get_csv_download_link(analyzed_df), unsafe_allow_html=True)
            
            # 分析統計を表示
            show_analysis_stats(analyzed_df)
            
            # 詳細な分析結果の表示
            st.subheader("発言内容の詳細分析")
            for idx, row in analyzed_df.iterrows():
                with st.expander(f"発言 {row['発言番号']} - {row['発言者']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**元の発言内容:**")
                        st.write(row['発言内容'])
                    with col2:
                        st.markdown("**分析済み発言内容:**")
                        st.write(row['分析済み発言内容'])
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")

elif use_sample:
    # サンプルデータを処理
    st.subheader("サンプルデータ")
    st.dataframe(sample_data)
    
    analyzed_df = process_csv(sample_data)
    if analyzed_df is not None:
        st.subheader("分析結果")
        st.dataframe(analyzed_df)
        st.markdown(get_csv_download_link(analyzed_df, "sample_analyzed_data.csv"), unsafe_allow_html=True)
        
        # 分析統計を表示
        show_analysis_stats(analyzed_df)
        
        # 詳細な分析結果の表示
        st.subheader("発言内容の詳細分析")
        for idx, row in analyzed_df.iterrows():
            with st.expander(f"発言 {row['発言番号']} - {row['発言者']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**元の発言内容:**")
                    st.write(row['発言内容'])
                with col2:
                    st.markdown("**分析済み発言内容:**")
                    st.write(row['分析済み発言内容'])
else:
    st.info("CSVファイルをアップロードするか、サンプルデータを使用してください。")

# ヘルプセクション
with st.expander("使い方ガイド"):
    st.markdown("""
    ## 授業記録分析ツールの使い方
    
    ### 基本的な使い方
    1. CSVファイルをアップロードするか、サンプルデータを使用します
    2. アプリケーションが自動的に発言内容を分析し、適切なブラケットを付けます
    3. 分析結果を確認し、必要に応じてダウンロードします
    
    ### ブラケットの種類
    - **[　]** - 具体的な事例（例：「例えば〜」「昨日〜」「〜さんは」など）
    - **（　）** - 教科・学習内容・社会的概念（概念辞書に基づく）
    - **〈　〉** - 児童生徒のアイデア・思い・構想（「思います」「考えます」など）
    
    ### カスタマイズ
    - **概念辞書のアップロード**: テキストファイルで概念辞書をカスタマイズできます
    - **パターン設定**: 具体例やアイデアを検出するためのパターンをカスタマイズできます
    - **分析オプション**: 文脈分析やブラケット重複時の戦略を設定できます
    
    ### 概念辞書のフォーマット
