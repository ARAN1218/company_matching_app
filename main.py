import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import os
# from PIL import Image
# LangChain & Google Generative AI 関連
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import google.api_core.exceptions
import gyoukai
import gyoukai_info
import slider_captions
industries_list = gyoukai.INDUSTRIES_LIST
industry_intros = gyoukai_info.INDUSTRIES_INFO
slider_captions = slider_captions.SLIDER_CAPTIONS

# .envファイルから環境変数を読み込む(テスト時のみ有効)
# from dotenv import load_dotenv
# load_dotenv()
# file_id = os.environ.get("COMPANY_FILE_ID")
# gemini_api_key = os.environ.get("GEMINI_API_KEY")

# .envファイルから環境変数を読み込む(本番時のみ有効)
file_id = st.secrets["COMPANY_FILE_ID"]
gemini_api_key = st.secrets["GEMINI_API_KEY"]


# アプリのタイトルと設定
# 画像も設定できる：favicon = Image.open("favicon.ico")
st.set_page_config(
    page_title="企業・業界マッチングアプリ",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title('🔍 企業・業界マッチングアプリ')
st.write('✨ あなたの希望条件に合った企業や業界を見つけましょう！')

# --- データ読み込みと前処理 ---
@st.cache_data
def load_data():
    # CSVファイルの読み込み
    try:
        # search result [1] を参照
        url = f"https://drive.google.com/uc?id={file_id}"
        df = pd.read_csv(url)
    except FileNotFoundError:
        st.error("エラー: データが読み込めませんでした")
        st.stop()

    # 比較に使用する特徴量
    features = ["待遇面の満足度","社員の士気","風通しの良さ","社員の相互尊重",
                "20代成長環境","人材の長期育成","法令順守意識","人事評価の適正感"]
    display_numerical_features = ['平均年収', '残業時間(月間)', '有給休暇消化率']

    # 欠損値処理 & データクリーニング
    df = df.dropna(subset=features)
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=features)
    for col in features:
        df = df[(df[col] >= 1) & (df[col] <= 5)]

    # 参考数値データ処理
    for col in display_numerical_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan

    # 企業名リストとデフォルト企業インデックス
    company_names_list = sorted(df['企業名'].unique().tolist())
    default_company_name = "デロイトトーマツリスクアドバイザリー合同会社"
    try:
        default_company_index = company_names_list.index(default_company_name)
    except ValueError:
        default_company_index = 0

    return df, features, display_numerical_features, company_names_list, default_company_index

df, features, display_numerical_features, company_names_list, default_company_index = load_data()

# 業界リスト
all_industries_options = ['全業界'] + industries_list
industries_for_analysis = industries_list

# --- 業界ごとの平均値計算 ---
@st.cache_data
def calculate_industry_averages(_df, _industries_for_analysis, _features, _display_numerical_features):
    industry_avg = {}
    # 全業界
    industry_avg['全業界'] = {}
    for feature in _features:
        industry_avg['全業界'][feature] = _df[feature].mean()
    for num_feature in _display_numerical_features:
        industry_avg['全業界'][num_feature] = _df[num_feature].mean(skipna=True)
    # 各業界
    for industry in _industries_for_analysis:
        industry_data = _df[_df['業界'] == industry]
        if not industry_data.empty:
            avg_values = {}
            for feature in _features:
                avg_values[feature] = industry_data[feature].mean()
            for num_feature in _display_numerical_features:
                 avg_values[num_feature] = industry_data[num_feature].mean(skipna=True)
            industry_avg[industry] = avg_values
    return industry_avg

industry_avg = calculate_industry_averages(df, industries_for_analysis, features, display_numerical_features)

# --- サイドバー ---
st.sidebar.header('🔧 共通条件設定')
app_mode = st.sidebar.radio(
    "📋 機能を選択",
    ["🔎 企業マッチング", "📊 業界マッチング", "🔄 企業比較", "🌐 業界比較", "📈 業界・企業分析"],
    key='app_mode_radio'
)

selected_industry_filter = st.sidebar.selectbox('絞り込み業界を選択（企業マッチング用）', all_industries_options, key='common_industry_filter')
st.sidebar.subheader('🎯 あなたの希望条件を入力してください (各1〜5点)')
st.sidebar.caption("各項目について、あなたが企業や業界に求める度合いを1(低い)〜5(高い)で評価してください。")
user_preferences = {}
for feature in features:
    user_preferences[feature] = st.sidebar.markdown(f"**{feature}**")  # タイトル
    st.sidebar.caption(slider_captions.get(feature, ""))  # キャプション（薄い文字）
    user_preferences[feature] = st.sidebar.slider(
        " ",  # ラベルは空にしてタイトルとキャプションだけ表示
        1.0, 5.0, 3.0, 0.1, key=f"common_{feature}"
    )

# --- 共通関数 ---
# レーダーチャート描画 (変更なし)
def plot_radar_chart(data_list, labels, chart_title, categories=None):
    if categories is None: categories = features
    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    valid_data_found = False
    for i, (data, label) in enumerate(zip(data_list, labels)):
        if data is None or len(data) != len(categories):
            st.warning(f"警告: 「{label}」のデータ形式が不正か、データが存在しません。グラフから除外します。")
            continue
        valid_data_found = True
        data_closed = list(data) + [data[0]]
        categories_closed = categories + [categories[0]]
        fig.add_trace(go.Scatterpolar(r=data_closed, theta=categories_closed, fill='toself', name=label, line_color=colors[i % len(colors)], opacity=0.6, hovertemplate='<b>%{theta}</b>: %{r:.2f}<extra></extra>'))
    if not valid_data_found:
        st.warning("グラフに表示できる有効なデータがありません。")
        return None
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[1, 5.1], tickvals=[1, 2, 3, 4, 5], tickfont=dict(size=10)), angularaxis=dict(tickfont=dict(size=11))), title=dict(text=chart_title, font=dict(size=16)), showlegend=True, legend=dict(font=dict(size=11), yanchor="bottom", y=-0.2, xanchor="center", x=0.5, orientation="h"), height=450, margin=dict(l=60, r=60, t=80, b=80))
    return fig

# ベクトル化関数 (変更なし)
def get_company_vector(company_data):
    if company_data is None: return None
    if isinstance(company_data, pd.Series): vector = [company_data.get(feature, 3.0) for feature in features]
    elif isinstance(company_data, dict): vector = [company_data.get(feature, 3.0) for feature in features]
    else: return None
    vector = [v if isinstance(v, (int, float)) and 1 <= v <= 5 else 3.0 for v in vector]
    return vector

def get_user_vector(user_prefs):
    if user_prefs is None: return None
    vector = [user_prefs.get(feature, 3.0) for feature in features]
    vector = [v if isinstance(v, (int, float)) else 3.0 for v in vector]
    return vector

def get_industry_vector(industry_avg_data):
    if industry_avg_data is None: return None
    vector = [industry_avg_data.get(feature, 3.0) for feature in features]
    vector = [v if pd.notna(v) and isinstance(v, (int, float)) else 3.0 for v in vector]
    return vector

# おすすめ度計算関数
def calculate_recommend_scores(user_vector, target_vectors):
    """
    ユーザーベクトルとターゲットベクトル群について、
    ①ユークリッド距離（0-50に正規化）
    ②コサイン類似度（0-50に正規化）
    を合算して100点満点のおすすめ度を計算
    """
    if user_vector is None or target_vectors is None or len(target_vectors) == 0:
        return np.array([])

    # ユークリッド距離計算
    user_vectors_repeated = np.tile(user_vector, (len(target_vectors), 1))
    euclidean_distances = np.linalg.norm(target_vectors - user_vectors_repeated, axis=1)

    # コサイン類似度計算
    cosine_similarities = cosine_similarity([user_vector], target_vectors).flatten()

    # ユークリッド距離を0-50に正規化（距離が小さいほど高スコア）
    if np.all(euclidean_distances == euclidean_distances[0]):
        euclidean_scores = np.full_like(euclidean_distances, 50.0)
    else:
        euclidean_scaler = MinMaxScaler(feature_range=(0, 50))
        euclidean_scores = 50.0 - euclidean_scaler.fit_transform(euclidean_distances.reshape(-1, 1)).flatten()

    # コサイン類似度を0-50に正規化
    if np.all(cosine_similarities == cosine_similarities[0]):
        cosine_scores = np.full_like(cosine_similarities, 50.0)
    else:
        cosine_scaler = MinMaxScaler(feature_range=(0, 50))
        cosine_scores = cosine_scaler.fit_transform(cosine_similarities.reshape(-1, 1)).flatten()

    # 合算（最大100点）
    recommend_scores = euclidean_scores + cosine_scores

    return recommend_scores

# --- Gemini 説明生成関数 (LCEL形式に修正) ---
@st.cache_data
def generate_company_descriptions(companies, api_key):
    """Gemini APIを使って複数の企業の説明文を一度に生成する (LCEL版)"""
    if not api_key:
        return ["APIキーが設定されていません."] * len(companies)

    try:
        # LLMの初期化
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, convert_system_message_to_human=False, google_api_key=api_key)

        # プロンプトテンプレート
        template = """
        あなたは就活のエキスパートです。与えられた企業情報をもとに、各企業への応募を検討している学生に向けて、80文字以内でその企業の魅力を伝えてください。
        {company_details}
        """
        # prompt = PromptTemplate(template=template, input_variables=["company_details"])
        prompt = ChatPromptTemplate.from_template(template)

        # LCELチェーン
        chain = prompt | llm | StrOutputParser()

        # 企業情報を整形
        company_details = ""
        for company in companies:
            company_scores = "\n".join([f"- {feat}: {company[feat]:.1f}点" for feat in features])
            company_details += f"企業名: {company['企業名']}\n業界: {company['業界']}\n評価スコア (5点満点):\n{company_scores}\n\n"

        # 呼び出し
        response = chain.invoke({"company_details": company_details})

        # レスポンスを企業ごとに分割
        # descriptions = response.strip().split("\n\n")
        return response

    except google.api_core.exceptions.PermissionDenied as e:
        st.error(f"APIキーが無効か、権限がありません: {e}")
        return ["APIキーが無効、またはAPIへのアクセス権限がありません."] * len(companies)
    except Exception as e:
        st.error(f"AIによる説明生成中にエラーが発生しました: {e}")
        return ["AIによる説明生成に失敗しました."] * len(companies)

def format_compare_value(company, item):
    val = company.get(item, np.nan);
    if pd.isna(val): return "N/A"
    if item == 'おすすめ度': return f"{val:.2f}"
    if item == '総合評価': return f"{val:.2f}"
    if item == '平均年収': return f"{val:.0f}万円"
    if item == '残業時間(月間)': return f"{val:.1f}時間"
    if item == '有給休暇消化率': return f"{val:.1f}%"
    if item in features: return f"{val:.2f}"
    return str(val)

def format_industry_compare_value(avg_data, item):
    if avg_data is None: return "N/A"
    value = avg_data.get(item, np.nan)
    if pd.isna(value): return "N/A"
    if item in features: return f"{value:.2f}"
    if item == '平均年収': return f"{value:.0f}万円"
    if item == '残業時間(月間)': return f"{value:.1f}時間"
    if item == '有給休暇消化率': return f"{value:.1f}%"
    return f"{value:.2f}"

# --- 機能ごとの実装 ---
# --- 企業マッチング機能 ---
if app_mode == "🔎 企業マッチング":
    st.header('🔎 企業マッチング')
    st.write("サイドバーで入力した希望条件に基づき、おすすめの企業を表示します。")

    # AI利用オプション
    use_ai_description = st.checkbox("AIによる企業紹介文を生成する", value=True, key='use_ai_description')

    if selected_industry_filter != '全業界':
        filtered_df = df[df['業界'] == selected_industry_filter].copy()
        st.write(f"**絞り込み業界:** {selected_industry_filter}")
        if filtered_df.empty:
            st.warning(f"「{selected_industry_filter}」業界に該当する企業データが見つかりませんでした。")
    else:
        filtered_df = df.copy()
        st.write(f"**絞り込み業界:** 全業界")
    search_button = st.sidebar.button('🔎 マッチング開始', key='match_company_button')

    # Gemini一括説明文生成
    def generate_bulk_company_descriptions(companies, api_key):
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, convert_system_message_to_human=False, google_api_key=api_key)
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            template = (
                """
                [システムプロンプト]
                あなたは就活コンサルタントのアイ先生です。これまで何人もの大卒就活生を自分に合った企業に合格させてきました。アイ先生の強みはその圧倒的企業分析力であり、企業の強みを要約して人に伝える事に関しては全人類の中でトップクラスの実力を持っています。
                
                [ユーザープロンプト]
                ＜指示＞
                以下の企業群の情報をもとに、各企業ごとに200文字程度で魅力を伝えてください。

                ＜制約＞
                ・紹介企業名を明示的に書かないで、説明だけ書いてください。
                ・各企業の紹介文は必ず半角のアットマーク「@」で区切って出力してください。全角のアットマークは絶対に使わないでください。
                ・企業紹介文の長さは大体200字程度でお願いします。多少前後しても構いませんが、少なすぎるのは情報が足りないと思いますのでご遠慮ください。
                ・与えられたデータ以外にも特筆すべきその企業独自の特徴等があれば是非書いてください。

                ＜出力テンプレート＞
                (企業①における200字程度の企業紹介文)@(企業②における200字程度の企業紹介文)@(企業③における200字程度の企業紹介文)@...

                ＜企業詳細データ＞
                {company_details}
                """
            )
            prompt = ChatPromptTemplate.from_template(template)
            company_details = ""
            for company in companies:
                company_scores = "\n".join([f"- {feat}: {company[feat]:.1f}点" for feat in features])
                company_details += f"企業名: {company['企業名']}\n業界: {company['業界']}\n評価スコア (5点満点):\n{company_scores}\n\n"
            chain = prompt | llm | StrOutputParser()
            response = chain.invoke({"company_details": company_details})
            return response.strip()
        except Exception as e:
            return f"AI説明文生成に失敗しました: {e}"

    if search_button and not filtered_df.empty:
        user_vector = get_user_vector(user_preferences)
        company_vectors = np.array([get_company_vector(row) for _, row in filtered_df.iterrows()])
        if user_vector is not None and len(company_vectors) > 0:
            recommend_scores = calculate_recommend_scores(user_vector, company_vectors)
        else:
            recommend_scores = np.array([])
        if len(recommend_scores) == len(filtered_df):
            result_df = filtered_df.copy()
            result_df['おすすめ度'] = recommend_scores
            result_df = result_df.sort_values(by='おすすめ度', ascending=False).head(10)
        else:
            st.error("おすすめ度の計算中にエラーが発生しました。")
            result_df = pd.DataFrame()

        # AI説明文一括生成と分割
        ai_descriptions = [""] * len(result_df)
        if use_ai_description and not result_df.empty:
            if not gemini_api_key:
                st.warning("環境変数にGEMINI_API_KEYが設定されていません。AIによる説明文生成はスキップされます。")
                ai_descriptions = ["AI説明文生成不可（APIキー未設定）"] * len(result_df)
            else:
                with st.spinner("AIが企業の特徴を分析中..."):
                    companies_data = result_df.to_dict('records')
                    bulk_text = generate_bulk_company_descriptions(companies_data, gemini_api_key)
                    # 「/」で分割し、余計な空白や改行を除去
                    ai_descriptions = [desc.strip() for desc in bulk_text.split("@") if desc.strip()]
                # 企業数と説明文数が一致しない場合は補正
                if len(ai_descriptions) != len(result_df):
                    # 足りない場合は空文字で補完
                    ai_descriptions += [""] * (len(result_df) - len(ai_descriptions))
                ai_descriptions = ai_descriptions[:len(result_df)]
        else:
            ai_descriptions = ["AIによる説明文生成はオフです。"] * len(result_df)

        # 結果データフレームにAI説明文列を追加
        if not result_df.empty:
            result_df = result_df.reset_index(drop=True)
            result_df['説明文'] = ai_descriptions

        if not result_df.empty:
            st.subheader('🏆 あなたにおすすめの企業トップ10')
            st.caption("あなたの希望条件と企業の評価の近さ（ユークリッド距離・コサイン類似度）に基づき、おすすめ度を0〜100点で表示しています（100点が最も近い）。")
            display_columns_match = ['企業名', '業界', '総合評価', 'おすすめ度'] + features
            st.dataframe(
                result_df[display_columns_match].style.format({
                    'おすすめ度': '{:.2f}',
                    '総合評価': '{:.2f}',
                    **{f: '{:.2f}' for f in features}
                }).background_gradient(subset=['おすすめ度'], cmap='viridis'),
                hide_index=True
            )
            st.caption("表の項目名をクリックすると並び替えができます（ただし表示はトップ10のみ）。")
            st.subheader('📊 マッチング企業のプロフィール詳細')
            num_companies_to_show = len(result_df)
            tab_labels = [f"{i+1}位: {result_df.iloc[i]['企業名']}" for i in range(num_companies_to_show)]
            tabs = st.tabs(tab_labels)
            for i, tab in enumerate(tabs):
                with tab:
                    company = result_df.iloc[i]
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown(f"#### 基本情報")
                        st.metric("おすすめ度", f"{company['おすすめ度']:.2f}")
                        st.markdown(f"**企業名:** {company['企業名']}")
                        st.markdown(f"**業界:** {company['業界']}")
                        st.markdown(f"**総合評価:** {company['総合評価']:.2f}")
                        avg_salary = company.get('平均年収', np.nan)
                        avg_overtime = company.get('残業時間(月間)', np.nan)
                        avg_vacation = company.get('有給休暇消化率', np.nan)
                        st.markdown(f"**平均年収:** {avg_salary:.0f}万円" if pd.notna(avg_salary) else "平均年収: データなし")
                        st.markdown(f"**残業時間:** {avg_overtime:.1f}時間/月" if pd.notna(avg_overtime) else "残業時間: データなし")
                        st.markdown(f"**有給休暇消化率:** {avg_vacation:.1f}%" if pd.notna(avg_vacation) else "有給消化率: データなし")
                        # 個別AI紹介文
                        st.markdown(f"**AIによる企業紹介:** {company.get('説明文', '説明文を生成できませんでした。')}")
                        if 'URL' in company and pd.notna(company['URL']):
                            st.link_button("OpenWorkで詳細を見る", company['URL'])
                    with col2:
                        user_vector_radar = get_user_vector(user_preferences)
                        company_vector_radar = get_company_vector(company)
                        if user_vector_radar and company_vector_radar:
                            radar_chart = plot_radar_chart(
                                [user_vector_radar, company_vector_radar],
                                ['あなたの希望', company['企業名']],
                                f"{company['企業名']} vs あなたの希望"
                            )
                            if radar_chart:
                                st.plotly_chart(radar_chart, use_container_width=True)
                        else:
                            st.warning("レーダーチャートを表示するためのデータが不足しています。")
                    st.markdown("---")
                    st.markdown("#### 評価項目の比較")
                    compare_items_data = {'項目': features}
                    compare_items_data['あなたの希望'] = [f"{user_preferences[f]:.1f}" for f in features]
                    compare_items_data[company['企業名']] = [f"{company.get(f, 'N/A'):.2f}" for f in features]
                    compare_items_df = pd.DataFrame(compare_items_data)
                    st.dataframe(compare_items_df.set_index('項目'), use_container_width=True)
        else:
            st.info("条件に合う企業が見つかりませんでした。")
        st.session_state['company_result_df'] = result_df if not result_df.empty else pd.DataFrame()
        st.session_state['user_prefs'] = user_preferences
    elif search_button and filtered_df.empty:
        pass
    else:
        st.info('👈 左側のサイドバーで希望条件を設定し、「マッチング開始」ボタンをクリックしてください。')


# --- 業界マッチング機能 (変更なし) ---
elif app_mode == "📊 業界マッチング":
    st.header("📊 業界マッチング")
    st.write("サイドバーで入力した希望条件に基づき、おすすめの業界を表示します。")
    search_industry_button = st.sidebar.button('📊 マッチング開始', key='match_industry_button')
    if search_industry_button:
        user_vector = get_user_vector(user_preferences)
        industry_vectors_dict = {name: get_industry_vector(avg_data) for name, avg_data in industry_avg.items()}
        valid_industry_names = [name for name, vec in industry_vectors_dict.items() if vec is not None]
        valid_industry_vectors = np.array([industry_vectors_dict[name] for name in valid_industry_names])
        if user_vector and len(valid_industry_vectors) > 0:
            recommend_scores_industry = calculate_recommend_scores(user_vector, valid_industry_vectors)
            scores_dict = {name: score for name, score in zip(valid_industry_names, recommend_scores_industry)}
        else:
            scores_dict = {}
        industry_match_data = []
        for industry_name in valid_industry_names:
            avg_data = industry_avg.get(industry_name, {})
            industry_match_data.append({
                '業界名': industry_name,
                'おすすめ度': scores_dict.get(industry_name, 0.0),
                **avg_data
            })
        if industry_match_data:
            result_industry_df = pd.DataFrame(industry_match_data)
            result_industry_df = result_industry_df.sort_values(by='おすすめ度', ascending=False).head(10).reset_index(drop=True)
            st.subheader('🏆 あなたにおすすめの業界トップ10')
            st.caption("あなたの希望条件と業界平均評価の近さ（ユークリッド距離・コサイン類似度）に基づき、おすすめ度を0〜100点で表示しています（100点が最も近い）。")
            display_columns_industry = ['業界名', 'おすすめ度'] + features
            st.dataframe(
                result_industry_df[display_columns_industry].style.format({
                    'おすすめ度': '{:.2f}',
                    **{f: '{:.2f}' for f in features}
                }).background_gradient(subset=['おすすめ度'], cmap='viridis'),
                hide_index=True
            )
            st.caption("表の項目名をクリックすると並び替えができます（ただし表示はトップ10のみ）。")
            st.subheader('📊 マッチング業界のプロフィール詳細')

            # --- AIによる業界紹介文をトップ10分まとめて一括生成 ---
            gemini_api_key = os.environ.get("GEMINI_API_KEY")
            ai_desc_list = [""] * len(result_industry_df)
            if gemini_api_key:
                try:
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash",
                        temperature=0.7,
                        convert_system_message_to_human=False,
                        google_api_key=gemini_api_key
                    )
                    from langchain_core.prompts import ChatPromptTemplate
                    from langchain_core.output_parsers import StrOutputParser
                    template = (
                        "あなたはキャリアアドバイザーです。以下の業界情報と、ユーザーの希望条件をもとに、"
                        "各業界がどれだけユーザーの希望に合っているかを評価しつつ、200字以内で業界の魅力を伝えてください。"
                        "各業界の紹介文は必ず半角のアットマーク「@」で区切って出力してください。\n"
                        "{industry_details}"
                    )
                    prompt = ChatPromptTemplate.from_template(template)
                    industry_details = ""
                    for idx, row in result_industry_df.iterrows():
                        industry_scores = "\n".join([f"- {feat}: {row.get(feat, 0):.1f}点" for feat in features])
                        user_scores = "\n".join([f"- {feat}: {user_preferences.get(feat, 0):.1f}点" for feat in features])
                        industry_details += (
                            f"業界名: {row['業界名']}\n"
                            f"業界スコア:\n{industry_scores}\n"
                            f"ユーザー希望:\n{user_scores}\n\n"
                        )
                    chain = prompt | llm | StrOutputParser()
                    with st.spinner("AIが業界の特徴を分析中..."):
                        ai_bulk = chain.invoke({"industry_details": industry_details})
                        ai_desc_list = [desc.strip() for desc in ai_bulk.split("@") if desc.strip()]
                    if len(ai_desc_list) != len(result_industry_df):
                        ai_desc_list += [""] * (len(result_industry_df) - len(ai_desc_list))
                    ai_desc_list = ai_desc_list[:len(result_industry_df)]
                except Exception as e:
                    ai_desc_list = [f"AI説明文生成に失敗: {e}"] * len(result_industry_df)
            else:
                ai_desc_list = ["AIによる業界紹介文生成はAPIキー未設定のため利用できません。"] * len(result_industry_df)

            # --- 各タブで個別にAI紹介文を表示 ---
            num_industries_to_show = len(result_industry_df)
            industry_tab_labels = [f"{i+1}位: {result_industry_df.iloc[i]['業界名']}" for i in range(num_industries_to_show)]
            industry_tabs = st.tabs(industry_tab_labels)
            for i, tab in enumerate(industry_tabs):
                with tab:
                    industry_info = result_industry_df.iloc[i]
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown(f"#### 基本情報 (平均値)")
                        st.metric("おすすめ度", f"{industry_info['おすすめ度']:.2f}")
                        st.markdown(f"**業界名:** {industry_info['業界名']}")
                        avg_salary = industry_info.get('平均年収', np.nan)
                        avg_overtime = industry_info.get('残業時間(月間)', np.nan)
                        avg_vacation = industry_info.get('有給休暇消化率', np.nan)
                        st.markdown(f"**平均年収:** {avg_salary:.0f}万円" if pd.notna(avg_salary) else "平均年収: データなし")
                        st.markdown(f"**残業時間:** {avg_overtime:.1f}時間/月" if pd.notna(avg_overtime) else "残業時間: データなし")
                        st.markdown(f"**有給休暇消化率:** {avg_vacation:.1f}%" if pd.notna(avg_vacation) else "有給消化率: データなし")
                        st.markdown(f"**AIによる業界紹介:** {ai_desc_list[i] if i < len(ai_desc_list) else '説明文を生成できませんでした。'}")
                    with col2:
                        user_vector_radar = get_user_vector(user_preferences)
                        industry_vector_radar = get_industry_vector(industry_info)
                        if user_vector_radar and industry_vector_radar:
                            radar_chart_industry = plot_radar_chart(
                                [user_vector_radar, industry_vector_radar],
                                ['あなたの希望', f"{industry_info['業界名']} (平均)"],
                                f"{industry_info['業界名']} 平均 vs あなたの希望"
                            )
                            if radar_chart_industry:
                                st.plotly_chart(radar_chart_industry, use_container_width=True)
                        else:
                            st.warning("レーダーチャートを表示するためのデータが不足しています。")
                    st.markdown("---")
                    st.markdown("#### 評価項目 (平均値)")
                    compare_industry_items_data = {'項目': features}
                    compare_industry_items_data['あなたの希望'] = [f"{user_preferences[f]:.1f}" for f in features]
                    compare_industry_items_data[f"{industry_info['業界名']} (平均)"] = [f"{industry_info.get(f, np.nan):.2f}" for f in features]
                    compare_industry_items_df = pd.DataFrame(compare_industry_items_data)
                    st.dataframe(compare_industry_items_df.set_index('項目'), use_container_width=True)
        else:
            st.info("条件に合う業界が見つかりませんでした。")
        st.session_state['industry_result_df'] = result_industry_df
        st.session_state['user_prefs'] = user_preferences
    else:
        st.info('👈 左側のサイドバーで希望条件を設定し、「マッチング開始」ボタンをクリックしてください。')

# --- 企業比較機能 (変更なし) ---
elif app_mode == "🔄 企業比較":
    st.header('🔄 企業比較')
    if 'company_result_df' not in st.session_state or st.session_state['company_result_df'].empty: st.warning("⚠️ まず「企業マッチング」機能で企業を検索し、比較したい企業候補を表示させてください。")
    else:
        result_df = st.session_state['company_result_df']
        user_prefs = st.session_state.get('user_prefs')
        st.subheader('比較する企業を選択')
        company_options = {f"{i+1}位 ({result_df.iloc[i]['おすすめ度']:.2f}): {result_df.iloc[i]['企業名']}": i for i in range(len(result_df))}
        col1, col2 = st.columns(2)
        with col1: selected_label1 = st.selectbox('1つ目の企業', company_options.keys(), index=0, key='compare_company1'); company1_idx = company_options[selected_label1]
        with col2: selected_label2 = st.selectbox('2つ目の企業', company_options.keys(), index=min(1, len(result_df)-1), key='compare_company2'); company2_idx = company_options[selected_label2]
        company1 = result_df.iloc[company1_idx]; company2 = result_df.iloc[company2_idx]
        st.subheader('📊 比較結果'); st.markdown("#### 主要情報と評価項目")
        compare_data_disp = {'項目': ['おすすめ度', '業界', '総合評価', '平均年収', '残業時間(月間)', '有給休暇消化率'] + features}
        
        compare_data_disp[company1['企業名']] = [format_compare_value(company1, item) for item in compare_data_disp['項目']]
        compare_data_disp[company2['企業名']] = [format_compare_value(company2, item) for item in compare_data_disp['項目']]
        show_user_in_table = st.checkbox("比較表にあなたの希望も表示", value=False, key='compare_table_show_user')
        if show_user_in_table and user_prefs:
            user_values = ["-", "-", "-"] + ["-", "-", "-"] + [f"{user_prefs[f]:.1f}" for f in features]
            compare_data_disp['あなたの希望'] = user_values
        compare_disp_df = pd.DataFrame(compare_data_disp); st.dataframe(compare_disp_df.set_index('項目'), use_container_width=True)
        st.subheader('📈 評価項目比較 (1-5点)'); vectors_to_plot = []; labels_to_plot = []
        company1_vector_radar = get_company_vector(company1); company2_vector_radar = get_company_vector(company2)
        if company1_vector_radar: vectors_to_plot.append(company1_vector_radar); labels_to_plot.append(company1['企業名'])
        if company2_vector_radar: vectors_to_plot.append(company2_vector_radar); labels_to_plot.append(company2['企業名'])
        show_user_in_radar = st.checkbox("グラフにあなたの希望も表示", value=True, key='compare_radar_show_user')
        if show_user_in_radar and user_prefs:
            user_vector_radar = get_user_vector(user_preferences);
        if user_vector_radar: vectors_to_plot.insert(0, user_vector_radar); labels_to_plot.insert(0, 'あなたの希望')
        if vectors_to_plot:
            radar_chart_compare = plot_radar_chart(vectors_to_plot, labels_to_plot, f"評価項目比較")
            if radar_chart_compare: st.plotly_chart(radar_chart_compare, use_container_width=True)
        else: st.warning("比較するデータがありません。")
        st.subheader('🔗 詳細情報 (OpenWork)'); col1_link, col2_link = st.columns(2)
        with col1_link:
            if 'URL' in company1 and pd.notna(company1['URL']): st.link_button(f"{company1['企業名']} のページへ", company1['URL'])
        with col2_link:
            if 'URL' in company2 and pd.notna(company2['URL']): st.link_button(f"{company2['企業名']} のページへ", company2['URL'])

# --- 業界比較機能 (変更なし) ---
elif app_mode == "🌐 業界比較":
    st.header("🌐 業界比較")
    st.write("選択した業界の平均データ（またはあなたの希望条件）を比較します。")
    st.subheader('比較する業界を選択')
    available_industries = ['全業界'] + industries_for_analysis
    col1, col2 = st.columns(2)
    with col1: selected_industry1 = st.selectbox("1つ目の業界", available_industries, index=0, key='compare_industry1')
    with col2: selected_industry2 = st.selectbox("2つ目の業界", available_industries, index=min(1, len(available_industries)-1), key='compare_industry2')
    industry1_avg_data = industry_avg.get(selected_industry1); industry2_avg_data = industry_avg.get(selected_industry2)
    st.subheader('📊 比較結果'); st.markdown("#### 評価項目と参考情報 (平均値)")
    compare_industry_data = {'項目': features + display_numerical_features}
    
    compare_industry_data[f"{selected_industry1} (平均)"] = [format_industry_compare_value(industry1_avg_data, item) for item in compare_industry_data['項目']]
    compare_industry_data[f"{selected_industry2} (平均)"] = [format_industry_compare_value(industry2_avg_data, item) for item in compare_industry_data['項目']]
    show_user_in_industry_table = st.checkbox("比較表にあなたの希望も表示", value=False, key='compare_industry_table_show_user')
    
    user_prefs = st.session_state.get('user_prefs')
    if show_user_in_industry_table and user_prefs:
        user_values = [f"{user_preferences[f]:.1f}" for f in features] + ["-", "-", "-"]
        compare_industry_data['あなたの希望'] = user_values
    compare_industry_df = pd.DataFrame(compare_industry_data); st.dataframe(compare_industry_df.set_index('項目'), use_container_width=True)
    st.subheader('📈 評価項目比較 (1-5点平均)'); industry_vectors_to_plot = []; industry_labels_to_plot = []
    industry1_vector_radar = get_industry_vector(industry1_avg_data); industry2_vector_radar = get_industry_vector(industry2_avg_data)
    if industry1_vector_radar: industry_vectors_to_plot.append(industry1_vector_radar); industry_labels_to_plot.append(f"{selected_industry1} (平均)")
    if industry2_vector_radar: industry_vectors_to_plot.append(industry2_vector_radar); industry_labels_to_plot.append(f"{selected_industry2} (平均)")
    show_user_in_industry_radar = st.checkbox("グラフにあなたの希望も表示", value=True, key='compare_industry_radar_show_user')
    if show_user_in_industry_radar: user_vector_radar = get_user_vector(user_preferences)
    if user_vector_radar: industry_vectors_to_plot.insert(0, user_vector_radar); industry_labels_to_plot.insert(0, 'あなたの希望')
    if industry_vectors_to_plot:
        radar_chart_industry_compare = plot_radar_chart(industry_vectors_to_plot, industry_labels_to_plot, f"業界評価項目比較")
        if radar_chart_industry_compare: st.plotly_chart(radar_chart_industry_compare, use_container_width=True)
    else: st.warning("比較する業界データがありません。")

# --- 業界・企業分析機能（レーダーチャート横並び修正版） ---
# --- 業界・企業分析機能（おすすめ度正規化＆業界紹介文全業界分） ---
elif app_mode == "📈 業界・企業分析":
    st.header('📈 業界・企業分析')
    st.write("サイドバーで設定したあなたの希望条件と、特定の業界平均または特定の企業を比較します。")
    user_vector_analysis = get_user_vector(user_preferences)
    if user_vector_analysis is None:
        st.error("ユーザー希望条件ベクトルを作成できませんでした。")
        st.stop()
    st.subheader('比較対象の選択')
    analysis_comparison_type = st.radio(
        "比較対象を選択してください", ["業界平均と比較", "特定の企業と比較"],
        key="analysis_comparison_type", horizontal=True
    )

    if analysis_comparison_type == "業界平均と比較":
        available_industries = ['全業界'] + industries_for_analysis
        selected_industry_analysis = st.selectbox('比較したい業界を選択', available_industries, key="analysis_industry_select")

        # --- おすすめ度を全業界で計算し、該当業界のみピックアップ ---
        industry_vectors_dict = {name: get_industry_vector(avg_data) for name, avg_data in industry_avg.items()}
        valid_industry_names = [name for name, vec in industry_vectors_dict.items() if vec is not None]
        valid_industry_vectors = np.array([industry_vectors_dict[name] for name in valid_industry_names])
        user_vec = get_user_vector(user_preferences)
        selected_score = None
        if user_vec is not None and len(valid_industry_vectors) > 0:
            recommend_scores = calculate_recommend_scores(user_vec, valid_industry_vectors)
            industry_score_dict = {name: score for name, score in zip(valid_industry_names, recommend_scores)}
            selected_score = industry_score_dict.get(selected_industry_analysis, None)

        industry_avg_data = industry_avg.get(selected_industry_analysis)
        industry_vector_analysis = get_industry_vector(industry_avg_data) if industry_avg_data else None

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"#### 基本情報 (平均値)")
            st.metric("おすすめ度", f"{selected_score:.2f}" if selected_score is not None else "N/A")
            st.markdown(f"**業界名:** {selected_industry_analysis}")
            st.markdown(f"**総合評価:** {industry_avg_data.get('総合評価', 'N/A'):.2f}" if industry_avg_data and '総合評価' in industry_avg_data else "総合評価: データなし")
            avg_salary = industry_avg_data.get('平均年収', np.nan) if industry_avg_data else np.nan
            avg_overtime = industry_avg_data.get('残業時間(月間)', np.nan) if industry_avg_data else np.nan
            avg_vacation = industry_avg_data.get('有給休暇消化率', np.nan) if industry_avg_data else np.nan
            st.markdown(f"**平均年収:** {avg_salary:.0f}万円" if pd.notna(avg_salary) else "平均年収: データなし")
            st.markdown(f"**残業時間:** {avg_overtime:.1f}時間/月" if pd.notna(avg_overtime) else "残業時間: データなし")
            st.markdown(f"**有給休暇消化率:** {avg_vacation:.1f}%" if pd.notna(avg_vacation) else "有給消化率: データなし")
            st.markdown(f"**AIによる業界紹介:** {industry_intros.get(selected_industry_analysis, '（紹介文未設定）')}")
        with col2:
            if industry_vector_analysis is not None:
                radar_chart_industry = plot_radar_chart(
                    [user_vector_analysis, industry_vector_analysis],
                    ['あなたの希望', f'{selected_industry_analysis} (平均)'],
                    f"{selected_industry_analysis} 平均 vs あなたの希望"
                )
                if radar_chart_industry:
                    st.plotly_chart(radar_chart_industry, use_container_width=True)
            else:
                st.warning("業界平均のベクトルを作成できませんでした。")
        st.markdown("---")
        st.markdown("#### 評価項目 (平均値)")
        if industry_avg_data:
            industry_info_disp = {
                '項目': features + display_numerical_features,
                '平均値': [format_industry_compare_value(industry_avg_data, f) for f in features]
                            + [format_industry_compare_value(industry_avg_data, nf) for nf in display_numerical_features]
            }
            st.dataframe(pd.DataFrame(industry_info_disp).set_index('項目'), use_container_width=True)
        else:
            st.error(f"「{selected_industry_analysis}」の平均データが見つかりません。")

    else:
        selected_company_analysis = st.selectbox('比較したい企業を選択', company_names_list, index=default_company_index, key="analysis_company_select")
        # --- おすすめ度を全企業で計算し、該当企業のみピックアップ ---
        user_vec = get_user_vector(user_preferences)
        company_vectors = np.array([get_company_vector(row) for _, row in df.iterrows()])
        all_company_names = df['企業名'].tolist()
        selected_score = None
        if user_vec is not None and len(company_vectors) > 0:
            recommend_scores = calculate_recommend_scores(user_vec, company_vectors)
            company_score_dict = {name: score for name, score in zip(all_company_names, recommend_scores)}
            selected_score = company_score_dict.get(selected_company_analysis, None)

        company_data_analysis_series = df[df['企業名'] == selected_company_analysis].iloc[0] if not df[df['企業名'] == selected_company_analysis].empty else None
        company_vector_analysis = get_company_vector(company_data_analysis_series) if company_data_analysis_series is not None else None

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"#### 基本情報")
            st.metric("おすすめ度", f"{selected_score:.2f}" if selected_score is not None else "N/A")
            if company_data_analysis_series is not None:
                st.markdown(f"**企業名:** {company_data_analysis_series['企業名']}")
                st.markdown(f"**業界:** {company_data_analysis_series['業界']}")
                st.markdown(f"**総合評価:** {company_data_analysis_series.get('総合評価', np.nan):.2f}")
                avg_salary = company_data_analysis_series.get('平均年収', np.nan)
                avg_overtime = company_data_analysis_series.get('残業時間(月間)', np.nan)
                avg_vacation = company_data_analysis_series.get('有給休暇消化率', np.nan)
                st.markdown(f"**平均年収:** {avg_salary:.0f}万円" if pd.notna(avg_salary) else "平均年収: データなし")
                st.markdown(f"**残業時間:** {avg_overtime:.1f}時間/月" if pd.notna(avg_overtime) else "残業時間: データなし")
                st.markdown(f"**有給休暇消化率:** {avg_vacation:.1f}%" if pd.notna(avg_vacation) else "有給消化率: データなし")
                # AIによる企業紹介（従来通り、必要なら生成）
                gemini_api_key = os.environ.get("GEMINI_API_KEY")
                ai_desc = ""
                if gemini_api_key:
                    with st.spinner("AIが企業の特徴を分析中..."):
                        desc_text = generate_company_descriptions([company_data_analysis_series], gemini_api_key)
                        if isinstance(desc_text, list) and len(desc_text) > 0:
                            ai_desc = desc_text[0].strip()
                        elif isinstance(desc_text, str):
                            ai_desc = desc_text.strip()
                st.markdown(f"**AIによる企業紹介:** {ai_desc if ai_desc else '説明文を生成できませんでした。'}")
                if 'URL' in company_data_analysis_series and pd.notna(company_data_analysis_series['URL']):
                    st.link_button("OpenWorkで詳細を見る", company_data_analysis_series['URL'])
        with col2:
            if company_vector_analysis is not None:
                radar_chart_company = plot_radar_chart(
                    [user_vector_analysis, company_vector_analysis],
                    ['あなたの希望', company_data_analysis_series['企業名']],
                    f"{company_data_analysis_series['企業名']} vs あなたの希望"
                )
                if radar_chart_company:
                    st.plotly_chart(radar_chart_company, use_container_width=True)
            else:
                st.warning("企業のベクトルを作成できませんでした。")
        st.markdown("---")
        st.markdown("#### 評価項目")
        if company_data_analysis_series is not None:
            company_info_disp = {
                '項目': features + display_numerical_features,
                '値': [f"{company_data_analysis_series.get(f, np.nan):.2f}" for f in features] +
                       [format_compare_value(company_data_analysis_series, nf) for nf in display_numerical_features]
            }
            st.dataframe(pd.DataFrame(company_info_disp).set_index('項目'), use_container_width=True)
        else:
            st.error(f"「{selected_company_analysis}」のデータが見つかりません。")

# --- フッター (変更なし) ---
st.markdown('---')
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8em;">
    © 2025 企業マッチングアプリ | このアプリケーションは教育およびデモンストレーション目的で作成されています。
</div>
""", unsafe_allow_html=True)
