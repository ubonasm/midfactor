import streamlit as st
import pandas as pd
import numpy as np
import spacy
import re
import json
import base64
from io import StringIO
import os
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import random

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
    df['分析済み発言内容'] = df['発言��容'].apply(analyze_text_with_context)
    
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

# 発言をブラケット種類別に分解する関数
def decompose_utterance(text):
    """発言をブラケット種類別に分解する"""
    # 最も外側のブラケットを優先して分解
    result = {
        "例示": [],  # [...]
        "概念": [],  # （...）
        "構想": [],  # 〈...〉
        "その他": []
    }
    
    # 正規表現パターン
    example_pattern = r"\[(.*?)\]"
    concept_pattern = r"（(.*?)）"
    idea_pattern = r"〈(.*?)〉"
    
    # 最も外側のブラケットを検出
    remaining_text = text
    
    # 例示ブラケット [...]
    example_matches = re.findall(example_pattern, remaining_text)
    for match in example_matches:
        result["例示"].append(match)
        remaining_text = remaining_text.replace(f"[{match}]", "", 1)
    
    # 概念ブラケット （...）
    concept_matches = re.findall(concept_pattern, remaining_text)
    for match in concept_matches:
        result["概念"].append(match)
        remaining_text = remaining_text.replace(f"（{match}）", "", 1)
    
    # 構想ブラケット 〈...〉
    idea_matches = re.findall(idea_pattern, remaining_text)
    for match in idea_matches:
        result["構想"].append(match)
        remaining_text = remaining_text.replace(f"〈{match}〉", "", 1)
    
    # 残りのテキストを「その他」に追加
    if remaining_text.strip():
        result["その他"].append(remaining_text.strip())
    
    return result

# マトリクス可視化用のデータを準備する関数
def prepare_matrix_data(df):
    """マトリクス可視化用のデータを準備する"""
    matrix_data = []
    
    for idx, row in df.iterrows():
        utterance_num = row.get('発言番号', idx + 1)
        speaker = row.get('発言者', 'Unknown')
        content = row['分析済み発言内容']
        
        # 発言をブラケット種類別に分解
        decomposed = decompose_utterance(content)
        
        # 各ブラケット種類ごとにデータを追加
        for bracket_type, texts in decomposed.items():
            if texts:  # 空でない場合のみ追加
                matrix_data.append({
                    "発言番号": utterance_num,
                    "発言者": speaker,
                    "ブラケット種類": bracket_type,
                    "テキスト数": len(texts),
                    "テキスト": " | ".join(texts[:3]) + ("..." if len(texts) > 3 else ""),
                    "全テキスト": " | ".join(texts)
                })
    
    return pd.DataFrame(matrix_data)

# マトリクス可視化を作成する関数
def create_matrix_visualization(matrix_df, selected_speakers=None):
    """マトリクス可視化を作成する"""
    if matrix_df.empty:
        # 空のデータフレームの場合は空のグラフを返す
        fig = go.Figure()
        fig.update_layout(
            title="データがありません",
            xaxis=dict(title="ブラケット種類"),
            yaxis=dict(title="発言番号"),
            height=400
        )
        return fig
    
    # 発言者が選択されていない場合は全ての発言者を選択
    if selected_speakers is None or len(selected_speakers) == 0:
        selected_speakers = matrix_df['発言者'].unique().tolist()
    
    # 発言者ごとに色を割り当て
    speakers = matrix_df['発言者'].unique()
    
    # 色のリストを作成（必要な数だけ）
    color_list = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.G10
    # 色が足りない場合はランダムな色を生成
    while len(color_list) < len(speakers):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        color_list.append(f"rgb({r},{g},{b})")
    
    # 色マップを作成
    color_map = {speaker: color_list[i % len(color_list)] for i, speaker in enumerate(speakers)}
    
    # 選択された発言者のデータのみをフィルタリング
    filtered_df = matrix_df.copy()
    
    # 選択された発言者とそれ以外で透明度を変える
    filtered_df['選択状態'] = filtered_df['発言者'].apply(
        lambda x: '選択済み' if x in selected_speakers else '非選択'
    )
    
    # マーカーサイズの調整
    max_size = filtered_df['テキスト数'].max() if not filtered_df.empty else 1
    filtered_df['マーカーサイズ'] = filtered_df['テキスト数'].apply(
        lambda x: max(10, min(30, 10 + (x / max_size) * 20))
    )
    
    # ホバーテキストの作成
    filtered_df['ホバーテキスト'] = filtered_df.apply(
        lambda row: f"発言番号: {row['発言番号']}<br>発言者: {row['発言者']}<br>ブラケット: {row['ブラケット種類']}<br>内容: {row['全テキスト']}",
        axis=1
    )
    
    # 発言者ごとのマーカーシンボルを設定
    marker_symbols = {speaker: idx for idx, speaker in enumerate(speakers)}
    filtered_df['マーカーシンボル'] = filtered_df['発言者'].map(marker_symbols)
    
    # 選択状態に応じたマーカーの設定
    filtered_df['マーカー表示'] = filtered_df['選択状態'].apply(
        lambda x: '🔸' if x == '選択済み' else '⚪'
    )
    
    # プロットの作成
    fig = go.Figure()
    
    # 非選択の発言者を先に追加（透明度を下げる）
    non_selected = filtered_df[filtered_df['選択状態'] == '非選択']
    for speaker in non_selected['発言者'].unique():
        speaker_df = non_selected[non_selected['発言者'] == speaker]
        
        # 色マップにない発言者の場合はデフォルト色を使用
        speaker_color = color_map.get(speaker, "gray")
        
        fig.add_trace(go.Scatter(
            x=speaker_df['ブラケット種類'],
            y=speaker_df['発言番号'],
            mode='markers+text',
            marker=dict(
                size=speaker_df['マーカーサイズ'],
                color=speaker_color,
                opacity=0.3,
                line=dict(width=0)
            ),
            text=speaker_df['マーカー表示'],
            textposition="middle center",
            name=speaker,
            hovertext=speaker_df['ホバーテキスト'],
            hoverinfo='text',
            showlegend=True
        ))
    
    # 選択された発言者を後に追加（通常の透明度）
    selected = filtered_df[filtered_df['選択状態'] == '選択済み']
    for speaker in selected['発言者'].unique():
        speaker_df = selected[selected['発言者'] == speaker]
        
        # 色マップにない発言者の場合はデフォルト色を使用
        speaker_color = color_map.get(speaker, "blue")
        
        fig.add_trace(go.Scatter(
            x=speaker_df['ブラケット種類'],
            y=speaker_df['発言番号'],
            mode='markers+text',
            marker=dict(
                size=speaker_df['マーカーサイズ'],
                color=speaker_color,
                opacity=1.0,
                line=dict(width=1, color='black')
            ),
            text=speaker_df['マーカー表示'],
            textposition="middle center",
            name=speaker,
            hovertext=speaker_df['ホバーテキスト'],
            hoverinfo='text',
            showlegend=True
        ))
    
    # レイアウトの設定
    fig.update_layout(
        title="発言内容のマトリクス可視化",
        xaxis=dict(
            title="ブラケット種類",
            categoryorder='array',
            categoryarray=['例示', '概念', '構想', 'その他']
        ),
        yaxis=dict(
            title="発言番号",
            autorange="reversed"  # 発言番号を上から下に表示
        ),
        height=max(500, min(1000, len(filtered_df['発言番号'].unique()) * 30)) if not filtered_df.empty else 400,
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode='closest',
        legend=dict(
            title="発言者",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

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
        st.sidebar.error(f"辞書の読み込み中にエラーが発生しました: {str(e)}")
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
        st.sidebar.error(f"パターン設定の読み込み中にエラーが発生しました: {str(e)}")

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
            st.metric("具体例 [　]", example_count, f"{example_count/total_utterances:.1%}" if total_utterances > 0 else "0%")
        with col3:
            st.metric("概念 （　）", concept_count, f"{concept_count/total_utterances:.1%}" if total_utterances > 0 else "0%")
        with col4:
            st.metric("アイデア 〈　〉", idea_count, f"{idea_count/total_utterances:.1%}" if total_utterances > 0 else "0%")
        
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
analyzed_df = None

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
        st.error(f"エラーが発生しました: {str(e)}")

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

# マトリクス可視化セクション
if analyzed_df is not None:
    st.header("マトリクス可視化")
    
    try:
        # マトリクスデータの準備
        matrix_df = prepare_matrix_data(analyzed_df)
        
        if not matrix_df.empty:
            # 発言者選択
            speakers = analyzed_df['発言者'].unique().tolist()
            selected_speakers = st.multiselect(
                "発言者を選択（複数選択可）",
                options=speakers,
                default=speakers
            )
            
            # マトリクス可視化の表示
            matrix_fig = create_matrix_visualization(matrix_df, selected_speakers)
            st.plotly_chart(matrix_fig, use_container_width=True)
            
            # マトリクスデータの詳細表示
            with st.expander("マトリクスデータの詳細"):
                st.dataframe(matrix_df)
        else:
            st.warning("マトリクス可視化用のデータがありません。")
    except Exception as e:
        st.error(f"マトリクス可視化中にエラーが発生しました: {str(e)}")
        st.info("デバッグ情報: マトリクスデータの準備または表示に問題があります。")

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
    
    ### マトリクス可視化の使い方
    1. 分析結果が表示された後、「マトリクス可視化」セクションが表示されます
    2. 発言者を選択すると、選択された発言者の発言がハイライト表示されます
    3. マトリクスでは、縦軸に発言番号、横軸にブラケット種類が表示されます
    4. 点の大きさは、その発言におけるブラケット種類の出現回数を表します
    5. 点にカーソルを合わせると、詳細情報が表示されます
    
    ### 概念辞書のフォーマット
    \`\`\`
    [カテゴリ名]
    用語1, 用語2, 用語3
    用語4, 用語5, 用語6
    
    [別のカテゴリ]
    用語A, 用語B, 用語C
    \`\`\`
    
    ### パターン設定のフォーマット
    \`\`\`
    # 具体例パターン
    パターン1
    パターン2
    
    # アイデア・思い・構想パターン
    パターン3
    パターン4
    \`\`\`
    """)

# フッター
st.markdown("---")
st.markdown("© 2025 授業記録分析ツール | マトリクス可視化機能付き")
