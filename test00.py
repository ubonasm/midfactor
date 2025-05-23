import streamlit as st
import pandas as pd
import spacy
import re
import json
import base64
from io import StringIO
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

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

def get_bracket_type(text):
    """テキストから最も外側のブラケットタイプを取得"""
    if text.startswith('[') and text.endswith(']'):
        return "例示"
    elif text.startswith('（') and text.endswith('）'):
        return "概念"
    elif text.startswith('〈') and text.endswith('〉'):
        return "構想"
    else:
        return "その他"

def decompose_utterance_by_brackets(text):
    """発言をブラケット種類別に分解"""
    # ブラケットで囲まれた部分を抽出
    bracket_segments = {
        "例示": [],
        "概念": [],
        "構想": [],
        "その他": []
    }
    
    # 各種ブラケットのパターンを検索
    example_matches = re.findall(r'\[([^\]]+)\]', text)
    concept_matches = re.findall(r'（([^）]+)）', text)
    idea_matches = re.findall(r'〈([^〉]+)〉', text)
    
    # ブラケットを除去したテキスト
    clean_text = text
    clean_text = re.sub(r'\[[^\]]+\]', '', clean_text)
    clean_text = re.sub(r'（[^）]+）', '', clean_text)
    clean_text = re.sub(r'〈[^〉]+〉', '', clean_text)
    clean_text = clean_text.strip()
    
    bracket_segments["例示"] = example_matches
    bracket_segments["概念"] = concept_matches
    bracket_segments["構想"] = idea_matches
    if clean_text:
        bracket_segments["その他"] = [clean_text]
    
    return bracket_segments

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

def create_matrix_visualization(df):
    """マトリクス可視化を作成"""
    if '分析済み発言内容' not in df.columns:
        return None
    
    # 発言をブラケット種類別に分解
    matrix_data = []
    for idx, row in df.iterrows():
        segments = decompose_utterance_by_brackets(row['分析済み発言内容'])
        
        for bracket_type, content_list in segments.items():
            for content in content_list:
                matrix_data.append({
                    '発言番号': row['発言番号'],
                    '発言者': row['発言者'],
                    'ブラケット種類': bracket_type,
                    '内容': content,
                    '発言順序': idx
                })
    
    if not matrix_data:
        return None
    
    matrix_df = pd.DataFrame(matrix_data)
    
    # ピボットテーブルを作成（発言ごとの各ブラケット種類の有無）
    pivot_df = matrix_df.groupby(['発言順序', '発言番号', '発言者', 'ブラケット種類']).size().reset_index(name='count')
    pivot_table = pivot_df.pivot_table(
        index=['発言順序', '発言番号', '発言者'], 
        columns='ブラケット種類', 
        values='count', 
        fill_value=0
    )
    
    # カラムの順序を指定
    column_order = ['例示', '概念', '構想', 'その他']
    existing_columns = [col for col in column_order if col in pivot_table.columns]
    pivot_table = pivot_table[existing_columns]
    
    return pivot_table, matrix_df

def plot_interactive_matrix(pivot_table, matrix_df, selected_speakers=None):
    """インタラクティブなマトリクス可視化を作成"""
    if pivot_table is None or matrix_df is None:
        return None
    
    # 発言者の色分け用のカラーマップを作成
    speakers = pivot_table.index.get_level_values('発言者').unique()
    colors = px.colors.qualitative.Set3[:len(speakers)]
    speaker_colors = dict(zip(speakers, colors))
    
    # 選択された発言者のハイライト
    if selected_speakers:
        highlight_mask = pivot_table.index.get_level_values('発言者').isin(selected_speakers)
    else:
        highlight_mask = [True] * len(pivot_table)
    
    # ヒートマップ用のデータを準備
    z_data = pivot_table.values
    y_labels = [f"発言{row[1]} ({row[2]})" for row in pivot_table.index]
    x_labels = pivot_table.columns.tolist()
    
    # 発言者別の色情報を準備
    speaker_info = [pivot_table.index[i][2] for i in range(len(pivot_table))]
    
    # カスタムカラースケールを作成
    fig = go.Figure()
    
    # 各発言者ごとに異なる色でヒートマップを作成
    for speaker in speakers:
        speaker_mask = [info == speaker for info in speaker_info]
        speaker_indices = [i for i, mask in enumerate(speaker_mask) if mask]
        
        if not speaker_indices:
            continue
        
        # 選択されているかどうかで透明度を調整
        opacity = 1.0 if not selected_speakers or speaker in selected_speakers else 0.3
        
        # 各ブラケット種類ごとに散布図を作成
        for j, bracket_type in enumerate(x_labels):
            for i in speaker_indices:
                value = z_data[i, j]
                if value > 0:
                    fig.add_trace(go.Scatter(
                        x=[j],
                        y=[i],
                        mode='markers',
                        marker=dict(
                            size=max(10, value * 20),  # 値に応じてサイズを調整
                            color=speaker_colors[speaker],
                            opacity=opacity,
                            line=dict(width=2, color='black' if speaker in (selected_speakers or []) else 'gray')
                        ),
                        name=speaker,
                        showlegend=speaker_indices[0] == i and j == 0,  # 最初の点のみ凡例に表示
                        hovertemplate=f"<b>{y_labels[i]}</b><br>" +
                                    f"ブラケット種類: {bracket_type}<br>" +
                                    f"出現回数: {value}<br>" +
                                    "<extra></extra>"
                    ))
    
    # レイアウトを設定
    fig.update_layout(
        title="発言内容のブラケット種類別マトリクス",
        xaxis=dict(
            title="ブラケット種類",
            tickmode='array',
            tickvals=list(range(len(x_labels))),
            ticktext=x_labels,
            side='top'
        ),
        yaxis=dict(
            title="発言（時系列順）",
            tickmode='array',
            tickvals=list(range(len(y_labels))),
            ticktext=y_labels,
            autorange='reversed'  # 上から下へ時系列順
        ),
        height=max(600, len(y_labels) * 25),
        width=800,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig

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

# サンプルデータの表示
st.sidebar.subheader("サンプルデータ形式")
sample_data = pd.DataFrame({
    '発言番号': [1, 2, 3, 4, 5, 6, 7, 8],
    '発言者': ['教師', '生徒A', '生徒B', '生徒C', '教師', '生徒A', '生徒D', '教師'],
    '発言内容': [
        '今日は三角形の面積について学びましょう。',
        '例えば、この図形の面積はどうやって求めますか？',
        '底辺×高さ÷2だと思います。',
        '昨日、お父さんと一緒に公園で三角形の看板を見ました。',
        'みなさんの考えを聞かせてください。',
        '私は、もっと簡単な方法があると思います。',
        '数学は面白いですね。',
        'それでは次の問題に進みましょう。'
    ]
})
st.sidebar.dataframe(sample_data)

# ファイルアップロード
uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])

# サンプルデータを使用するオプション
use_sample = st.checkbox("サンプルデータを使用")

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
            
            # マトリクス可視化
            st.subheader("📊 マトリクス可視化")
            
            # 発言者選択機能
            if '発言者' in analyzed_df.columns:
                all_speakers = analyzed_df['発言者'].unique().tolist()
                selected_speakers = st.multiselect(
                    "ハイライトする発言者を選択（複数選択可）",
                    options=all_speakers,
                    default=None,
                    help="選択した発言者の発言がハイライトされます。何も選択しない場合は全ての発言者が表示されます。"
                )
                
                if not selected_speakers:
                    selected_speakers = None
            else:
                selected_speakers = None
            
            # マトリクス可視化を作成
            pivot_table, matrix_df = create_matrix_visualization(analyzed_df)
            
            if pivot_table is not None:
                # インタラクティブなマトリクス図を表示
                matrix_fig = plot_interactive_matrix(pivot_table, matrix_df, selected_speakers)
                if matrix_fig:
                    st.plotly_chart(matrix_fig, use_container_width=True)
                
                # マトリクスデータの詳細表示
                with st.expander("マトリクスデータの詳細"):
                    st.subheader("ブラケット種類別の発言分解")
                    
                    # フィルタリング機能
                    if selected_speakers:
                        filtered_matrix_df = matrix_df[matrix_df['発言者'].isin(selected_speakers)]
                    else:
                        filtered_matrix_df = matrix_df
                    
                    # ブラケット種類でフィルタリング
                    bracket_filter = st.selectbox(
                        "ブラケット種類でフィルタリング",
                        options=['全て'] + filtered_matrix_df['ブラケット種類'].unique().tolist()
                    )
                    
                    if bracket_filter != '全て':
                        filtered_matrix_df = filtered_matrix_df[filtered_matrix_df['ブラケット種類'] == bracket_filter]
                    
                    st.dataframe(filtered_matrix_df)
                    
                    # 統計情報
                    st.subheader("ブラケット種類別統計")
                    bracket_stats = filtered_matrix_df.groupby('ブラケット種類').size().reset_index(name='出現回数')
                    st.bar_chart(bracket_stats.set_index('ブラケット種類'))
            else:
                st.warning("マトリクス可視化用のデータが見つかりませんでした。")
            
            # 詳細な分析結果の表示
            st.subheader("発言内容の詳細分析")
            for idx, row in analyzed_df.iterrows():
                # 選択された発言者のハイライト
                is_highlighted = selected_speakers is None or row['発言者'] in selected_speakers
                
                with st.expander(f"{'🔸' if is_highlighted else '⚪'} 発言 {row['発言番号']} - {row['発言者']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**元の発言内容:**")
                        st.write(row['発言内容'])
                    with col2:
                        st.markdown("**分析済み発言内容:**")
                        st.write(row['分析済み発言内容'])
                    
                    # ブラケット分解の表示
                    segments = decompose_utterance_by_brackets(row['分析済み発言内容'])
                    st.markdown("**ブラケット種類別分解:**")
                    for bracket_type, content_list in segments.items():
                        if content_list:
                            st.markdown(f"- **{bracket_type}**: {', '.join(content_list)}")
                            
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
        
        # マトリクス可視化
        st.subheader("📊 マトリクス可視化")
        
        # 発言者選択機能
        if '発言者' in analyzed_df.columns:
            all_speakers = analyzed_df['発言者'].unique().tolist()
            selected_speakers = st.multiselect(
                "ハイライトする発言者を選択（複数選択可）",
                options=all_speakers,
                default=None,
                help="選択した発言者の発言がハイライトされます。何も選択しない場合は全ての発言者が表示されます。"
            )
            
            if not selected_speakers:
                selected_speakers = None
        else:
            selected_speakers = None
        
        # マトリクス可視化を作成
        pivot_table, matrix_df = create_matrix_visualization(analyzed_df)
        
        if pivot_table is not None:
            # インタラクティブなマトリクス図を表示
            matrix_fig = plot_interactive_matrix(pivot_table, matrix_df, selected_speakers)
            if matrix_fig:
                st.plotly_chart(matrix_fig, use_container_width=True)
            
            # マトリクスデータの詳細表示
            with st.expander("マトリクスデータの詳細"):
                st.subheader("ブラケット種類別の発言分解")
                
                # フィルタリング機能
                if selected_speakers:
                    filtered_matrix_df = matrix_df[matrix_df['発言者'].isin(selected_speakers)]
                else:
                    filtered_matrix_df = matrix_df
                
                # ブラケット種類でフィルタリング
                bracket_filter = st.selectbox(
                    "ブラケット種類でフィルタリング",
                    options=['全て'] + filtered_matrix_df['ブラケット種類'].unique().tolist()
                )
                
                if bracket_filter != '全て':
                    filtered_matrix_df = filtered_matrix_df[filtered_matrix_df['ブラケット種類'] == bracket_filter]
                
                st.dataframe(filtered_matrix_df)
                
                # 統計情報
                st.subheader("ブラケット種類別統計")
                bracket_stats = filtered_matrix_df.groupby('ブラケット種類').size().reset_index(name='出現回数')
                st.bar_chart(bracket_stats.set_index('ブラケット種類'))
        else:
            st.warning("マトリクス可視化用のデータが見つかりませんでした。")
        
        # 詳細な分析結果の表示
        st.subheader("発言内容の詳細分析")
        for idx, row in analyzed_df.iterrows():
            # 選択された発言者のハイライト
            is_highlighted = selected_speakers is None or row['発言者'] in selected_speakers
            
            with st.expander(f"{'🔸' if is_highlighted else '⚪'} 発言 {row['発言番号']} - {row['発言者']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**元の発言内容:**")
                    st.write(row['発言内容'])
                with col2:
                    st.markdown("**分析済み発言内容:**")
                    st.write(row['分析済み発言内容'])
                
                # ブラケット分解の表示
                segments = decompose_utterance_by_brackets(row['分析済み発言内容'])
                st.markdown("**ブラケット種類別分解:**")
                for bracket_type, content_list in segments.items():
                    if content_list:
                        st.markdown(f"- **{bracket_type}**: {', '.join(content_list)}")
else:
    st.info("CSVファイルをアップロードするか、サンプルデータを使用してください。")
