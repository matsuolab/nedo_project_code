# %%
"""
instruciton datasetの生成

概要
- ハルシネーションを含むinst data
- codeや数学類は､最小限

"""
import pandas as pd
import os
from datasets import load_dataset
import json
import random
from tqdm import tqdm
import re

data_folder="data/0524with_halcination_little_codes"

#dataフォルダ内をリセット
os.system(f"mkdir {data_folder}")
os.system(f"rm -rf {data_folder}/*")


ds_dict={}


# %%
# cleaning script
def clean_autogen(text):
    if text is None:
        return ""
    text=text.strip()
    return text

question_template="以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n"
answer_template="\n\n### 応答:\n"


records=[]

# # mixtralで自動生成したQ&A

score_threshold=4
ng_words=[
          #回答を避けるプロンプトの削除
          "申し訳","分からない","分かりません","すみません",
          #図表などへの言及
          #"図","表",
          #"市","県",
          #"漢字",
          ]
#ハルシネーションも許容する
"""
#日本関係の事項はハルシネーションが多い
ng_words+= [
    "北海", "青森", "岩手", "宮城", "秋田", "山形", "福島",
    "茨城", "栃木", "群馬", "埼玉", "千葉", "東京", "神奈川",
    "新潟", "富山", "石川", "福井", "山梨", "長野", "岐阜",
    "静岡", "愛知", "三重", "滋賀", "京都", "大阪", "兵庫",
    "奈良", "和歌山", "鳥取", "島根", "岡山", "広島", "山口",
    "徳島", "香川", "愛媛", "高知", "福岡", "佐賀", "長崎",
    "熊本", "大分", "宮崎", "鹿児島", "沖縄"
]
ng_words+= [
    "札幌", "仙台", "新潟", "横浜", "川崎", "相模原", "名古屋",
    "京都", "大阪", "神戸", "堺", "広島", "北九州", "福岡",
    "熊本", "那覇", "千葉", "さいたま", "静岡", "浜松", "岡山",
    "鹿児島", "長崎", "松山", "高松", "徳島", "高知", "大分",
    "宮崎", "長野", "金沢", "富山", "福井", "松本", "豊橋",
    "岐阜", "甲府", "奈良", "和歌山", "姫路", "福山", "呉",
    "倉敷", "宇部", "下関", "高岡", "今治", "東広島", "伊勢",
    "旭川", "函館", "室蘭", "帯広", "釧路", "青森", "八戸",
    "弘前", "秋田", "盛岡", "山形", "福島", "郡山", "いわき",
    "宇都宮", "前橋", "高崎", "水戸", "日立", "土浦", "柏",
    "松戸", "浦安", "市川", "船橋", "川口", "越谷", "草加",
    "所沢", "川越", "熊谷", "相馬", "大宮", "高槻", "豊中",
    "茨木", "八尾", "寝屋川", "枚方", "平塚", "藤沢", "厚木",
    "横須賀", "三浦", "鎌倉", "小田原", "逗子", "鎌倉", "茅ヶ崎",
    "三鷹", "八王子", "立川", "府中", "調布", "町田", "稲城",
    "横浜", "厚木", "大和", "綾瀬", "藤沢", "鎌倉", "平塚",
    "鶴岡", "酒田", "米沢", "会津若松", "白河", "前橋", "草津",
    "沼津", "富士", "清水", "掛川", "豊田", "岡崎", "一宮",
    "津", "四日市", "鈴鹿", "名張", "甲賀", "彦根", "大津",
    "舞鶴", "長浜", "高槻", "枚方", "東大阪", "西宮", "芦屋",
    "加古川", "明石", "洲本", "淡路", "津山", "玉野", "赤磐",
    "出雲", "松江", "大田", "浜田", "益田", "鳥取", "倉吉",
    "米子", "西条", "新居浜", "大洲", "八幡浜", "今治", "佐世保",
    "平戸", "諫早", "雲仙", "南島原", "宇和島", "八幡浜", "中津",
    "豊後高田", "別府", "佐伯", "延岡", "小林", "日向", "串間",
    "都城", "曽於", "奄美", "名瀬", "日南", "伊佐", "薩摩川内"
]
ng_words+=[
    "寿司", "天ぷら", "ラーメン", "うどん", "そば", "抹茶", "お茶", "和牛", "刺身", "味噌",
    "醤油", "出汁", "海苔", "焼き鳥", "たこ焼き", "お好み焼き", "納豆", "梅干し", "御節料理", "懐石料理",
    "茶道", "華道", "書道", "剣道", "柔道", "空手", "相撲", "合気道", "弓道", "盆栽",
    "歌舞伎", "能", "狂言", "文楽", "雅楽", "俳句", "短歌", "浮世絵", "陶芸", "漆芸","寺",
    "折り紙", "切り紙", "漫画", "アニメ", "ゲーム", "カラオケ", "パチンコ", "温泉", "祭り", "花火",
    "桜", "紅葉", "富士山", "神社", "寺", "御守り", "御朱印", "着物", "浴衣", "下駄",
    "祭り", "神輿", "太鼓", "扇子", "団扇", "風鈴", "畳", "ふすま", "障子", "縁側",
    "庭園", "苔庭", "枯山水", "茶室", "茶会", "詩吟", "剣舞", "能楽", "能面", "狂言面",
    "和装", "浴衣", "羽織", "帯", "袴", "足袋", "草履", "雪駄", "忍者", "侍",
    "鎧", "刀", "武士道", "茶道具", "書道具", "筆", "墨", "硯", "和紙", "掛け軸",
    "日本酒", "焼酎", "泡盛", "梅酒", "甘酒", "酢", "味醂", "干物", "漬物", "佃煮",
    "和菓子", "羊羹", "団子", "饅頭", "大福", "最中", "おかき", "煎餅", "葛切り", "寒天",
    "金平糖", "和三盆", "葛餅", "茶碗蒸し", "茶粥", "釜飯", "炊き込みご飯", "味噌汁", "吸い物", "煮物",
    "焼き物", "揚げ物", "蒸し物", "酢の物", "和え物", "刺身", "寿司", "天ぷら", "かき揚げ", "たたき",
    "しゃぶしゃぶ", "すき焼き", "おでん", "肉じゃが", "お好み焼き", "たこ焼き", "焼きそば", "餃子", "ラーメン", "うどん",
    "そば", "お茶漬け", "煮込み", "鍋物", "うなぎ", "かば焼き", "山菜", "海藻", "昆布", "わかめ",
    "ひじき", "もずく", "くろも", "青のり", "おかゆ", "おにぎり", "巻き寿司", "押し寿司", "ちらし寿司", "稲荷寿司",
    "太巻き", "細巻き", "巻き物", "茶道", "華道", "書道", "剣道", "柔道", "空手", "合気道",
    "相撲", "弓道", "薙刀", "砲術", "居合道", "柔術", "槍術", "剣術", "古武道", "武道",
    "忍術", "忍者", "侍", "刀", "鎧", "兜", "武士道", "戦国時代", "江戸時代", "明治時代",
    "大正時代", "昭和時代", "平成時代", "令和時代", "平安時代", "鎌倉時代", "南北朝時代", "室町時代", "安土桃山時代", "幕末",
    "幕府", "将軍", "大名", "藩", "城", "町人", "農民", "商人", "武士", "刀鍛冶",
    "忍び装束", "槍", "薙刀", "弓", "弓術", "射箭", "武術", "弓術", "長刀", "手裏剣",
    "くノ一", "隠密", "間者", "諜報", "密偵", "間者", "忍び", "変わり身", "火遁", "水遁",
    "風遁", "土遁", "雷遁", "分身", "分身の術", "影分身", "影分身の術", "影", "手裏剣術", "投擲",
    "手甲", "鎖鎌", "十手", "鎖", "鎖術", "護身術", "護身", "護衛", "警護", "護法",
    "護符", "魔除け", "神道", "仏教", "禅", "浄土宗", "天台宗", "真言宗", "日蓮宗", "浄土真宗",
    "曹洞宗", "臨済宗", "黄檗宗", "修験道", "山岳信仰", "神社", "仏閣", "寺院", "庵", "道場",
    "参拝", "祈願", "供養", "祭祀", "奉納", "奉仕", "神道行事", "仏教行事", "葬儀", "法事",
    "儀式", "式典", "祭典", "神楽", "祭り", "祭礼", "祇園祭", "天神祭", "三社祭", "神嘗祭",
    "新嘗祭", "御霊祭", "節分", "雛祭り", "端午の節句", "七夕", "お盆", "お彼岸", "お正月", "初詣",
    "大晦日", "除夜の鐘", "除夜", "紅白歌合戦", "年越しそば", "新年", "年賀状", "お年玉", "初日の出", "初夢",
    "初売り", "福袋", "鏡開き", "松の内", "七草粥", "成人の日", "節分", "ひな祭り", "春分の日", "ゴールデンウィーク",
    "こどもの日", "母の日", "父の日", "海の日", "山の日", "敬老の日", "体育の日", "文化の日", "勤労感謝の日", "天皇誕生日",
    "祝日", "国民の祝日", "休日", "連休", "休暇", "夏休み", "冬休み", "春休み", 
    "祝祭日", "祝賀", "祭典", "行事", "催し物",  "大会", "式典", 
    "公演", "演奏会", "発表会", "展示会", "展覧会", "美術展", "写真展", "工芸展", "博覧会", "文化祭",
    "学園祭", "音楽祭", "映画祭", "演劇祭", "演劇", "舞台", "劇場", "歌劇", "音楽", "楽器",
    "歌手", "声優", "俳優", "女優", "監督", "脚本家", "プロデューサー", "映画", "テレビ", "ドラマ",
    "アニメ", "漫画", "イラスト", "アーティスト", "デザイナー", "イラストレーター", "画家", "写真家", "作家", "小説家",
]
"""
ng_words=list(set(ng_words))
#mixtralで自動生成された回答の一部はハルシネーションが起きやすいので､ルールベースで除外する
def is_probably_halcinated(a):

    #ハルシネーションで多いパターンを除外
    for ng_word in ng_words:
        if a.find(ng_word)>=0:
            return True
    return False
    """
    #日付のハルシネーションも許容する
    #年号などもハルシネーションの原因になる
    pattern = r"(\d{4}年|明治\d{1,2}年|大正\d{1,2}年|昭和\d{1,2}年|平成\d{1,2}年|令和\d{1,2}年|\d{1,2}月\d{1,2}日)"
    # マッチするかどうかをチェック
    matches = re.findall(pattern, a)
    if matches:
        return True
    
    #日付
    pattern = r'^\d{1,2}月\d{1,2}日$'
    if re.match(pattern, a):
        return True
    return False
    """

# %%
#書き出し､書き終わりの指示のチェック
q="「にすぎない」で終わる文章にしてください電界の概念は、マイケル・ファラデーが提唱したものである。電場は、帯電した物体がその周囲の空間に作り出すもので、電場の中に置かれた他の電荷に力を及ぼす。電場は2つの電荷の間に作用し、重力が2つの質量の間に作用するのと同じように、無限大に広がり、距離と逆二乗の関係を示します。電場は一般に空間的に変化し、ある1点におけるその強さは、静止した無視できる電荷がその点に置かれた場合に感じるであろう力（単位電荷あたり）と定義される。試験電荷」と呼ばれる概念的な電荷は、それ自身の電界が主電界を乱すのを防ぐために限りなく小さくなければならず、また磁界の影響を防ぐために静止していなければならない。"
q="「される」で終わる文章にしてください朝、ズボンが見つからない男は急いで自宅を飛び出し、隣人の宅に飛び込みます。そこで隣人に、急いでいると transmit される一方、不意に拾った古い写真集を開き、中身を見つめてしまいます。"
#q="「し」で始まる文に変えてください。コビレゴンドウの寿命は？"
a="寿命を知りたいのは、コビレゴンドウです。"
#q="aa"
a="以上が電場の概要です。詳細は割愛させていただきます。"
a="しじ"
#a="なされる"
#"

def check_command(q,a):
    #文末系の指示を守っているかどうかの判定
    pattern = r'^「(.*?)」で終わる'
    matches = re.findall(pattern,q)
    if matches:
        match_str=matches[0]
        if a.find(match_str)>=0:
            return True
        else:
            return False
        
    #文頭系の指示の判定
    pattern = r'^「(.*?)」で始まる'
    matches = re.findall(pattern,q)
    if matches:
        match_str=matches[0]
        if a.startswith(match_str):
            return True
        else:
            return False

    #指示が無い場合はTrue
    return True

print(check_command(q,a))


# %%
text_list=[
    "五城目テレビ中継局は1957年3月1日に設立され、東京都江戸川区五城目に位置しています。この地域の生活文化を取り上げる番組が多く制作されており、特に「江戸川区民と一緒に」などの番組は人気です。五城目テレビ中継局は地域の生活文化を取り上げることが特徴であり、現在も多くの人から支持されています。",
    "スヌープ・ドッグ風に1946年の映画『偉大なる期待』を2段落で要約できますか？",
    "3月1日",
    "３月1日",
    "3月はいい天気",
]
for text in text_list:
    print(is_probably_halcinated(text),text)

# %%
#マルチターン
ds=load_dataset("llm-jp/oasst1-21k-ja",)["train"]

records=[]
for record in tqdm(ds):
    conversations=record["conversations"]
    if len(conversations)!=4:
        continue

    #マルチターンで適切に2ターン目をマスキングする方法がよくわからないので､とりあえず､1ターン目は改行を少なくしておく.
    q=conversations[0]["value"]
    q+="\n### 応答:\n"+conversations[1]["value"]+"\n### 指示:\n"+conversations[2]["value"]
    a=conversations[3]["value"]
    text=f"{question_template}{q}{answer_template}{a}"
    records.append(text)

ds_dict["multi_turn_oasst"]=records
print(text),len(records)

# %%

#マルチターン oasst 英語
ds=load_dataset("llm-jp/oasst2-33k-en",)["train"]

records=[]
for record in tqdm(ds):
    conversations=record["conversations"]
    if len(conversations)!=4:
        continue
    q=conversations[0]["content"]
    q+="\n### 応答:\n"+conversations[1]["content"]+"\n### 指示:\n"+conversations[2]["content"]
    a=conversations[3]["content"]
    text=f"{question_template}{q}{answer_template}{a}"

    records.append(text)
ds_dict["multi_turn_oasst_en"]=records
print(text),len(records)

# %%
qa_list=[]

# %%
# マルチターン　自動生成

ds=load_dataset("kanhatakeyama/AutoMultiTurnByMixtral8x22b",)["train"]
ds.shuffle()


n_multi_turn=2*10**7 #自動生成マルチターンは、回答が短いケースが多いので、少なめにしておく

records=[]
cnt=0
for record in tqdm(ds):
    if cnt>=n_multi_turn:
        break
    #turn 1
    q=record["q1"]
    a=record["a1"]

    #短い回答は使わない｡ mtbenchは長文の方がスコアが上がりやすいため､
    if len(a)<10:
        continue

    if not is_probably_halcinated(a) and check_command(q,a):
    #if not is_probably_halcinated(a) and not is_probably_halcinated(q):
        text=f"{question_template}{q}{answer_template}{a}"
        cnt+=1
        records.append(text)
        #qa_list.append(hash)

    #turn 2
    q=record["q1"]
    q+="\n### 応答:\n"+record["a1"]+"\n### 指示:\n"+record["q2"]
    a=record["a2"]
    if len(a)<10:
        continue


    #if not is_probably_halcinated(a) and not is_probably_halcinated(q):
    if not is_probably_halcinated(a) and check_command(q,a):
        text=f"{question_template}{q}{answer_template}{a}"
        cnt+=1
        records.append(text)


ds_dict["multi_turn_mixtral"]=records
print(text)
print(len(records))

# %%
print(records[10])

# %%

exclude_count=0

datasets=[
    load_dataset("hatakeyama-llm-team/AutoGeneratedJapaneseQA",split="train"),
    load_dataset("hatakeyama-llm-team/AutoGeneratedJapaneseQA-CC",split="train"),
    load_dataset("hatakeyama-llm-team/AutoGeneratedJapaneseQA-other",split="train"),
    load_dataset("kanhatakeyama/OrcaJaMixtral8x22b",split="train"),
    load_dataset("kanhatakeyama/ChatbotArenaJaMixtral8x22b",split="train"),
    load_dataset("kanhatakeyama/LogicalDatasetsByMixtral8x22b",split="train"),
]

# %%


for dataset in datasets:
    dataset.shuffle()
    for original_record in tqdm(iter(dataset)):


        q=clean_autogen(original_record["question"])
        a=clean_autogen(original_record["answer"])
        if q=="" or a=="":
            continue


        if "score" in original_record:
            if original_record["score"] is None:
                continue
            if int(original_record["score"])<score_threshold:
                continue

        if is_probably_halcinated(a):
            continue
        if not check_command(q,a):
            continue
        #if is_probably_halcinated(q):
        #    continue

        text=f"{question_template}{q}{answer_template}{a}"
        #hash=hash_text(text)

        if a!="" and q!="":
            records.append(text)
            #qa_list.append(hash)
        
ds_dict["auto_gen_mixtral"]=records

print(len(records))
print(text)

# %%
print(records[-1000])

# %%

# %% [markdown]
# # hachiさんのalpaca + mixtral dataset

# %%

hachi_datasets={
   "HachiML/Hachi-Alpaca": load_dataset("HachiML/Hachi-Alpaca",split='v1.0_cleaned'),
    "HachiML/Evol-Alpaca-gen3-500":load_dataset("HachiML/Evol-Alpaca-gen3-500",split='train'),
    "HachiML/Evol-hh-rlhf-gen3-1k":load_dataset("HachiML/Evol-hh-rlhf-gen3-1k",split='train'),
    #"HachiML/alpaca_jp_math":load_dataset("HachiML/alpaca_jp_math",split='v1.0_cleaned'),
    #数学は、全くできない上に、pythonを使って解くことすらできない
}

# %%
records=[]
for key,hachi_ds in hachi_datasets.items():
    for record in tqdm(hachi_ds):
        q=record["instruction"]
        if "input" in record:
            inp=record["input"]
        else:
            inp=""
        if inp!="":
            q+="\n"+inp
        a=record["output"]
        if q=="" or a=="":
            continue
        text=f"{question_template}{q}{answer_template}{a}"
        records.append(text)

    ds_dict[f"hachi_{key}"]=records


# %%

# %% [markdown]
# # Bumpo dataset

#文法理解に関するデータセット
ds2=load_dataset("hatakeyama-llm-team/BumpoRikai",split="train")
# %%
records=[]
for original_record in tqdm(iter(ds2)):
    q=(original_record["question"])
    a=(original_record["answer"])
    inst=(original_record["instruction"])
    q=inst+"\n"+q
    if q=="" or a=="":
        continue
    text=f"{question_template}{q}{answer_template}{a}"
    records.append(text)
ds_dict["bumpo_rikai"]=records



# %%

#minnade
m_ds=load_dataset("minnade/chat-daily",split="train")

id_to_content={}
for record in m_ds:
    id_to_content[record["id"]]=record["body"]

questions=[]
for record in m_ds:
    if record["role"]=="assistant":
        try:
            q=id_to_content[record["parent_id"]]
        except:
            print("error: ",record)
        a=record["body"]
        if a is None:
            continue
        if len(a)<4:
            continue
        #questions.append((q,a))
        text=f"{question_template}{q}{answer_template}{a}"
        records.append(text)
ds_dict["minnade"]=questions

# %% [markdown]
# # code dataset

# %%
code_ds_dict={}

# %%
#openmathinst 数GB
# すべての問題を､pythonで解きましょう､みたいなデータセットで､逆効果っぽいのでやめる｡(モデルの数学/code性能が低いと､ただのゴミ出力になるだけ)
"""
openmath_ds=load_dataset("kunishou/OpenMathInstruct-1-1.8m-ja",split="train")

records=[]
for original_record in iter(openmath_ds):
    q=(original_record["question_ja"])
    a=(original_record["generated_solution_ja"])
    #inst=(original_record["instruction"])
    if q=="" or a=="":
        continue
    text=f"{question_template}{q}{answer_template}{a}"
    records.append(text)
code_ds_dict["openmathja"]=records
records[1]
"""

# %%
#code 50k
ds=load_dataset("HachiML/alpaca_jp_python",split="v1.0_cleaned")

records=[]
for original_record in iter(ds):
    q=(original_record["instruction"])
    a=(original_record["output"])
    inp=(original_record["input"])
    if inp!="":
        q+="\n"+inp
    if q=="" or a=="":
        continue
    text=f"{question_template}{q}{answer_template}{a}"
    records.append(text)
code_ds_dict["code_hachi"]=records



# %%
#code 5k
ds=load_dataset("kunishou/amenokaku-code-instruct",split="train")

records=[]
for original_record in iter(ds):
    q=(original_record["instruction"])
    a=(original_record["output"])
    inp=(original_record["input"])
    if inp!="":
        q+="\n"+inp
    if q=="" or a=="":
        continue
    text=f"{question_template}{q}{answer_template}{a}"
    records.append(text)
code_ds_dict["amenokaku"]=records



# %%
#算数 0.5k
ds=load_dataset("saldra/sakura_japanese_dataset",split="train")

records=[]
for original_record in iter(ds):
    q=(original_record["instruction"])
    a=(original_record["output"])
    inp=(original_record["input"])
    if inp!="":
        q+="\n"+inp
    if q=="" or a=="":
        continue
    text=f"{question_template}{q}{answer_template}{a}"
    records.append(text)
code_ds_dict["sakura"]=records



# %%
# meta math
#数学的な基礎力が低く、勉強してもあまり意味がなさそう
"""
ds=load_dataset("meta-math/MetaMathQA",split="train")

records=[]
for original_record in iter(ds):
    q=(original_record["query"])
    a=(original_record["response"])
    if q=="" or a=="":
        continue
    text=f"{question_template}{q}{answer_template}{a}"
    records.append(text)
code_ds_dict["meta_math"]=records
records[1]

"""

# %%
"""
ds=load_dataset("microsoft/orca-math-word-problems-200k",split="train")

records=[]
for original_record in iter(ds):
    q=(original_record["question"])
    a=(original_record["answer"])
    if q=="" or a=="":
        continue
    text=f"{question_template}{q}{answer_template}{a}"
    records.append(text)
code_ds_dict["orca_math"]=records
records[1]
"""


# %%

#python codes 150k
ds=load_dataset("m-a-p/CodeFeedback-Filtered-Instruction",split="train")

records=[]
for original_record in iter(ds):
    q=(original_record["query"])
    a=(original_record["answer"])
    if q=="" or a=="":
        continue
    text=f"{question_template}{q}{answer_template}{a}"
    records.append(text)
code_ds_dict["codefeedback"]=records



# %%
# 英語系　じどうせいせい 200万件ほど

ds=load_dataset("TIGER-Lab/WebInstructSub",split="train")


# %%
#ncライセンスのデータを除く
ds=ds.filter(lambda example: (example["source"])!="socratic")

# %%
#使うかどうかはケースバイケースとする
print(len(ds))
records=[]
for original_record in iter(ds):
    break
    q=(original_record["question"])
    a=(original_record["answer"])
    if q=="" or a=="":
        continue
    text=f"{question_template}{q}{answer_template}{a}"
    records.append(text)

code_ds_dict["webinstructsub"]=records



# %%
#四則演算 100万件
#使うかどうかはケースバイケースとする
ds=load_dataset("kanhatakeyama/Sansu",split="train")

records=[]
for original_record in iter(ds):
    break
    q=(original_record["question"])
    a=(original_record["answer"])
    if q=="" or a=="":
        continue
    text=f"{question_template}次の四則演算を計算しなさい｡\n{q}{answer_template}{a}"
    records.append(text)

code_ds_dict["shisoku"]=records
len(records)#,records[1]





# %%
# %%
all_code_records=[]
for k,v in code_ds_dict.items():
    all_code_records+=v
print("codes: ",len(all_code_records))

# %%



# %%
all_recrds=[]
for k,v in ds_dict.items():
    all_recrds+=v

# %%

def write_jsonl(records,
    output_path="data/all.jsonl",
    n_eval=500,
    n_train=10**10,
    ):

    print("shuffle...")
    random.shuffle(records)
    print("write...")
    df=pd.DataFrame()
    df["text"] =records[:-n_eval][:n_train]
    df["text"]=df["text"].astype(str)
    df=df.reset_index()
    df.to_parquet(output_path)
    
    #eval
    df=pd.DataFrame()
    df["text"] =records[-n_eval:]
    df["text"]=df["text"].astype(str)
    df=df.reset_index()
    df.to_parquet(output_path.replace(".parquet","_eval.parquet"))
    return df


# %%
print("dedup...")
integ_records=list(set(all_recrds+all_code_records))
_=write_jsonl(integ_records,f"{data_folder}/inst_data.parquet")

# %%
print("all: ",len(integ_records))

# %%
_#.to_csv("a.csv")

# %%



