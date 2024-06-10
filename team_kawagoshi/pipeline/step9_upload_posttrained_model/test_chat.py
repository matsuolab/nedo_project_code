import argparse
import os
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tokenizer_and_model_dir", type=str, required=True,)
    args = parser.parse_args()
    print(f"{args = }")
    return args


def load_tokenizer_and_model(input_tokenizer_and_model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(input_tokenizer_and_model_dir)
    model = AutoModelForCausalLM.from_pretrained(input_tokenizer_and_model_dir, device_map="auto",
                                                torch_dtype=torch.bfloat16)
    #model.generation_config.repetition_penalty = 1.2
    #print(model.generation_config.repetition_penalty)
    return tokenizer, model


def test_tokenizer_and_model(tokenizer, model, prompt_text: str) -> str:
    encoded_prompt_text = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        encoded_generation_text = model.generate(encoded_prompt_text, max_new_tokens=500, temperature=0.1, do_sample=True)[0]
    decoded_generation_text = tokenizer.decode(encoded_generation_text)
    return decoded_generation_text


def main() -> None:
    args = parse_arguments()

    test_prompt1 = """以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

### 指示:
ソクラテスは彼の時代の主流の考えにどのように挑戦しましたか？

### 応答:
"""

    test_prompt2 = """以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

### 指示:
質問に対する答えを出力してください。答えが複数の場合、コンマ（,）で繋げてください。

### 入力:
質問：日本の初代首相は？

### 応答:
"""

    test_prompt3 = """以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

### 指示:
与えられたテキストから同一の対象を指し示すフレーズを全て抽出してください。回答は以下のような形式で答えてください。
フレーズ1 フレーズ2 フレーズ3
フレーズ4 フレーズ5

### 入力:
就業規則とは、企業において使用者が労働基準法等に基づき、当該企業における労働条件等に関する具体的細目について定めた規則集のことをいう。

### 応答:
"""
    test_prompt4 = """以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。:

### 指示: 1 + 1300を求めなさい。 

### 応答:
"""

    test_prompt5 = """以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n### 指示:\nヴァージン・オーストラリア航空はいつから運航を開始したのですか？\n\n### 入力:\nヴァージン・オーストラリア航空（Virgin Australia Airlines Pty Ltd）はオーストラリアを拠点とするヴァージン・ブランドを冠する最大の船団規模を持つ航空会社です。2000年8月31日に、ヴァージン・ブルー空港として、2機の航空機、1つの空路を運行してサービスを開始しました。2001年9月のアンセット・オーストラリア空港の崩壊後、オーストラリアの国内市場で急速に地位を確立しました。その後はブリスベン、メルボルン、シドニーをハブとして、オーストラリア国内の32都市に直接乗り入れるまでに成長しました。\n\n### 応答:\n"""
    test_prompt6 = """以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n### 指示:\nロッククライミングで使用されるハーネスの種類をリストアップしてください。\n\n### 入力:\nクライミングの種類によって、ハーネスに求められる機能は異なります。スポーツクライミングでは、最小限のハーネスを使用し、ギアループを縫い付けているものもあります。アルパインクライマーは、軽量なハーネスを選ぶことが多く、レッグループが取り外し可能なものもあります。ビッグウォールクライマーは、パッド入りのウエストベルトとレッグループを好みます。また、骨盤が狭く、標準的なハーネスを安全に支えることができない子供用のフルボディハーネスもあります。倒立しても落下しないように、子供用として製造されたものや、ウェビングで作られたものがあります。倒立の可能性があるときや、重いザックを背負うときは、全身ハーネスを使用する人もいます。また、シットハーネスと組み合わせて使用するチェストハーネスもあります。UIAAの試験結果によると、チェストハーネスはシットハーネスよりも首への衝撃が少なく、フルボディハーネスと同じ利点があります[6]。\n\nこれらのハーネスとは別に、ケイビングハーネスやキャニオニングハーネスがあり、それぞれ異なる用途で使用されています。例えば、ケイビング用ハーネスは、防水でパッドなしの丈夫な素材でできており、2つのアタッチメントポイントがあります。この取り付け部分からマイヨンを離すと、ハーネスが素早く緩みます。\n\nキャニオニング用ハーネスは、クライミング用ハーネスに似ていて、パッドはありませんが、シートプロテクターが付いていて、懸垂下降をより快適に行うことができます。これらのハーネスは通常、ダイニーマの1つのアタッチメントポイントを持っています。### 応答:\n"""
    test_prompt7 = """Q: 日本の首都は? A:"""
    test_prompt8 = """以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

### 指示:アメリカの住宅所有者にはどのような税制上の優遇措置がありますか？

### 応答:アメリカでは、住宅所有者は主に以下の2つの税制上の優遇措置を受けることができます。ひとつは、住宅ローン利息 deduction（住宅ローンの利子が控除できること）、もう一つは、property tax deduction（所得税における不動産税の控除）です。

### 指示:住宅ローン利息 deductionについてもう少し詳しく教えてください。

### 応答:
"""
    test_prompt9 = """以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

### 指示:いつから人類は西暦を採用するようになったのでしょうか？

### 応答:人類が西暦を採用するようになったのは、紀元前6世紀のドロミティアヌス産まで遡ります。その時代、ポペオルニウス1世はローマ帝国を統治しており、彼が採用した年号計算法が西暦に発展しました。

### 指示:西暦の採用はどうやって普及したのですか？

### 応答:
"""
    test_prompt10 = """以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

### 指示:ドラえもんの「のび太」になりきって会話を始めましょう。では以下の質問から始めてください：""手を洗った後、エアドライヤーは必要だと思いますか？""

### 応答:
"""
    test_prompt11 = """以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

### 指示:美術の名作を子供向けのインタラクティブな体験に変えるためのアイデアを5つ挙げ、それぞれの作品とそのアイデアを説明してください。

### 応答:
"""

    test_prompt11 = """以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

### 指示:次の文章を要約してください。
「取材内容を記事にするとき、これでよいのか自信がない」

「顧客とのやり取りを要約する際、何から手を付けるのか分からない」

要約とは、元の文章から要点を抽出し自分の言葉でまとめることです。情報を選定することで、相手に届ける情報の密度を上げる行為ともいえます。要約が役立つ場面は、オンライン会議の内容伝達、業務マニュアルの作成、社内研修などさまざま。

今回は取材ライターとして活動する筆者が、要約の質が驚くほど研ぎ澄まされる11のコツをまとめました。要約スキルはちょっとしたコツを押さえるだけで改善できますので、ぜひ一緒に確認していきましょう。

要約とは？
要約とは文章のテーマや筆者の伝えたいことを、書き手（話し手）が解釈して簡潔にまとめること。

突然ですが「エレベーターピッチ」という言葉を聞いたことがありますか？エレベーターピッチとはエレベーターに乗っている15〜30秒ほどの時間で相手にプレゼンをすること。では、5000文字の文章の内容を上司に伝えるなら、何を伝えるでしょうか？これが「要約」です。

元の文章の構成や論理構造を崩さずにまとめる点がポイントで、文章全体の趣旨を伝えることが目的です。重要なキーワードを拾ってまとめていますので、元の文章より文字数は減っているのに情報の密度は高くなります。「誰が」「何を」「どうしたのか」といった5W1Hを意識すると話の核を捉えやすくなるでしょう。

要約には下記のようなメリットがあります。

元の文章への理解が深まる

テーマを分かりやすく伝えられる

必要な情報を拾いたい場合にも便利

相手の記憶に残る情報共有ができる

それぞれの情報の特徴や相違点に気付きやすくなる

相手に重要な点を効率的に伝えられるためプレゼン力が上がる

要約・要旨・要点の違い
要約と要旨の違い
要約	要旨
意味	論旨や要点をまとめること	筆者が言いたいことをまとめること
文章の長さ	元の文章に依存する	1文でもOK
文章の入れ替え	×	◯

要旨とは要点をかいつまんで、筆者が伝えたいことをまとめることです。言い換えれば、筆者の主張を伝えられるなら1文でも成り立ちます。

要約と要点の違い
要約	要点
意味	論旨や要点をまとめること	文章の大切な点
文章の長さ	元の文章に依存する	1文でもOK
主語と述語の追加・省略	△	◯

要点とは文章や段落の重要な部分のこと。要約が文章全体に対するポイントの抜き出しだとしたら、要点はさらに狭い範囲を指すものになります。

要約の書き方の手順とコツ
要約の書き方の手順を押さえることは、情報を適切に処理するのに大切。大まかな手順は、インプット（文章を把握する）→アウトプット（内容をまとめる）という感じです。また、事前に要約の文字数が指定されている場合は、文字数も確認しておくようにしましょう。

STEP1. 文章の全体像を把握する
まずは文章の全体像を把握しましょう。意味がわからないところがないか確認します。このとき、文章の内容を一言一句確認する必要はありません。もし余裕がある場合は、筆者の主張を自分の言葉に置き換えて伝えられるかと意識しながら読み進めてください。

STEP2. 文章を意味のかたまりに分ける
文章の全体像が把握できたら、今度は文章を意味のかたまりに分けてください。意味をまとめてかたまりとしているものを「意味段落」といいますが、この意味段落を自分で作っていくイメージです。このとき「しかし」「一方」「ところで」などの接続詞に注目すると作業がしやすくなります。

STEP3. 要点とキーワードを抽出する
続いて、要点とキーワードを抽出しましょう。この意味段落を一言で表すなら？これが要点です。キーワードは、筆者の主張・理由・結論に共通して含まれているワード。最後に答え合わせをしましょう。キーワードを順番通りにつなげて要点と一致する、全体として論理的な流れになっていればOKです。

STEP4. まとめる
最後にSTEP3で拾いあげた要点とキーワードを参考に、自分の言葉で内容をまとめましょう。専門用語は分かりやすく言い換え、補足説明をします。意味が似ている文章がある場合はどちらかを省きましょう。主語と述語がねじれていないか確認することも大切です。

要約のコツと注意点11選
受験対策として小論文の要約が求められたり、社会人としてビジネス文書の要約が求められたり。要約はさまざまな場面で必要とされています。要約のコツは、自分が伝えたいことではなく、元の文章の筆者が伝えたいことに焦点を当てること。ここでは、要約のコツや注意点を紹介していきます。

要約のコツ1.元の文章をそのまま使わない
要約において元の文章をそのまま引用するのは控えましょう。要約では筆者の主張を噛み砕いて自分の言葉で伝えることが重要だからです。また、無断で元の文章を使用してしまうと著作権法に抵触してしまう可能性もあります。

要約のコツ2.あいまいな表現は使わない
長ければ長いほどよいのではなく、短ければ短いほど価値が上がるのが要約文の魅力です。

そのため、「〜だと思います」「〜という感じがします」などあいまいな表現は使わず、短い言葉で断定的に言い切るようにしましょう。あいまいな表現では主張がぼやけてしまいますし、誤解を与え認識の相違を起こしてしまうからです。

要約のコツ3.文章の順番を入れ替えない
要約では文章の順番を入れ替えてはいけません。例えば筆者が例を挙げたあとに主張をしているのに、要約では主張が先に来ている…というパターンはNG。見落としがちなポイントですが、要約はあくまで論理構成を含めた「まとめ」なので、筆者の論理構成を崩さないようにしましょう。

要約のコツ4.PREP法を用いて要点を整理する
PREP法とは「結論」「理由」「具体例」「結論」から成る文章のフレームワークです。この型に当てはめることで、説得力を持って簡潔に情報を伝えられます。ただし、要約では理論構成を変えることはできませんので、あくまでもPREPの要素を含める、というスタンスで理解してください。

要約のコツ5.タイトルは要旨を端的に表している
タイトルには筆者が伝えたいポイントが凝縮されています。タイトルが設定されている文章の場合は見逃さないようにしましょう。意外かもしれませんが、要約力は読解力と密接な関係があります。タイトルから筆者が何を伝えたいのか、本文と照らし合わせて観察してください。また、見出しも本文の内容を要約している場合が多いです。タイトルと同じくらい見出しにも注目しましょう。

要約のコツ6.長くなってしまう場合は情報を抽象化する
文字制限があるのに要約文が長くなってしまう場合は、情報を抽象化しましょう。例えば、文字起こしアプリの機能として「PDFやDOCS、TXT、EXCEL、SRT、WAVでエクスポートできる」と書いた場合、「さまざまな（抽象化）形式でエクスポートできる」と書くことで文字数を削減できます。

要約のコツ7.全体像を把握して自分の言葉でまとめる
全体像を把握するための前提知識と情報収集は大切です。要約者が全体像を把握することで、読み手も話の道筋を掴みやすくなります。また、自分の言葉でまとめることも大切。その際は、筆者の主張に自分の思想が反映されないよう、中立的な表現を心がけましょう。

要約のコツ8.たとえを上手に活用して読者を惹き込む
たとえの使い方は、読者の理解度に影響を与えます。元の文章で筆者がたとえを使っている場合は、要約に活用できないか確認してみましょう。ただし要約の文字制限がある場合はたとえはカットした方がよい場合もあります。

要約のコツ9.キーワードをきちんと含める
要約とは「要するに〜」の後にくる内容を指すため、キーワードが必ず含まれます。キーワードを見つけるコツは、繰り返し出てくる言葉を探すことと、鍵かっこで括られている言葉を探すことです。また、固有名詞は具体的な情報である可能性が高いので見逃さないようにしましょう。

要約のコツ10.読者が全体像を予想できるよう心がける
読み手が全体像を正しくつかむには、要約の中に筆者の「主張・主張の理由・具体例・結論」が含まれていることが大切です。数字情報は具体的な根拠や主張を支える説得力になりますので、見逃さないようにしてください。映画や小説は先の見えない展開にドキドキ・ハラハラするものですが、要約の場合はその反対で、先がスムーズに読めることが大切です。

要約のコツ11.筆者の主張・結論・理由・事実を含める
筆者の主張・結論・理由・事実は基本的に漏らさずに含めましょう。情報を適切に整理して並べることは、相手に理解されやすくなるだけでなく、情報を取りやすくすることでもあります。ちなみに要約の場合はあくまで筆者の考えをまとめることなので、行間を読む必要はなく、書かれている内容を総括すればOKです。

コツと一緒に確認したいオンライン会議における要約の必要性
ZOOMやMicrosoft Teamsなどで行われるオンライン会議の議事録を取って、その内容を要約したい方もいるでしょう。ここでは、オンライン会議における要約の必要性を解説します。

必要な情報を効率的に共有できる
オンライン会議を要約すれば、情報をコンパクトにラッピングし相手に届けられます。例えばオンライン会議に参加していなかった上司に「会議どうだった？」と聞かれた場合、「〜という課題が挙げられ、〜という解決策が提案されました。今後は〜という方向で意見がまとまり、来週の会議で〜を部署ごとに報告する予定です」と答えられれば、要点を押さえた効率的な情報共有ができるでしょう。

やるべき課題を簡潔に伝えられる
議事録の目的は共通認識を確認して、やるべき課題を整理することです。議事録全体に目を通しつつ、実行すべきタスクを確認できるのも、オンライン会議における要約のメリット。簡単な記録として要約内容を残しておくだけでも、オンライン会議後の事後評価に役立ちます。

業務パフォーマンスを向上できる
要約では冗長表現が省かれるため、必要な情報を効率的に共有できます。1時間の議事録では文字数が1万文字を超える場合がほとんどであるため、議事録を読むだけでも10〜20分の時間がかかってしまいます。要約して共有することで要点だけをサッと伝えることができ、業務効率化につながるでしょう。

要約の効率を上げるコツはAIを活用すること
これまで要約の手順や要約するときのコツを紹介してきましたが、いざ要約をしようとすると、文章に目を通して、キーセンテンスに付箋を貼って…とやることが多くて大変ですよね。そこでおすすめしたいのが、文字起こしアプリNottaのChatGPTによるAI要約機能です。

NottaはAI音声認識エンジンを搭載した文字起こしアプリ。高度な技術が実装されているため1時間ほどの音声なら5分ほどで処理できますし、日常会話であれば8割以上の精度で出力できます。複数のフォーマットで出力でき、編集機能もあるため複数のソフトを往復する必要もありません。
### 応答:
"""
    # Loads and tests the local tokenizer and the local model.
    local_tokenizer, local_model = load_tokenizer_and_model(args.input_tokenizer_and_model_dir)
    local_decoded_generation_text = test_tokenizer_and_model(local_tokenizer, local_model, test_prompt1)
    local_decoded_generation_text2 = test_tokenizer_and_model(local_tokenizer, local_model, test_prompt2)
    local_decoded_generation_text3 = test_tokenizer_and_model(local_tokenizer, local_model, test_prompt3)
    local_decoded_generation_text4 = test_tokenizer_and_model(local_tokenizer, local_model, test_prompt4)
    local_decoded_generation_text5 = test_tokenizer_and_model(local_tokenizer, local_model, test_prompt5)
    local_decoded_generation_text6 = test_tokenizer_and_model(local_tokenizer, local_model, test_prompt6)
    local_decoded_generation_text7 = test_tokenizer_and_model(local_tokenizer, local_model, test_prompt7)
    local_decoded_generation_text8 = test_tokenizer_and_model(local_tokenizer, local_model, test_prompt8)
    local_decoded_generation_text9 = test_tokenizer_and_model(local_tokenizer, local_model, test_prompt9)
    local_decoded_generation_text10 = test_tokenizer_and_model(local_tokenizer, local_model, test_prompt10)
    local_decoded_generation_text11 = test_tokenizer_and_model(local_tokenizer, local_model, test_prompt11)
    # Checks the generated text briefly.
    print()
    print(f"{test_prompt1 = }")
    print(f"{local_decoded_generation_text = }")
    print()
    print()
    print(f"{test_prompt2 = }")
    print(f"{local_decoded_generation_text2 = }")
    print()
    print(f"{test_prompt3 = }")
    print(f"{local_decoded_generation_text3 = }")
    print()
    print(f"{test_prompt4 = }")
    print(f"{local_decoded_generation_text4 = }")
    print()
    print(f"{test_prompt5 = }")
    print(f"{local_decoded_generation_text5 = }")
    print()
    print(f"{test_prompt6 = }")
    print(f"{local_decoded_generation_text6 = }")
    print()
    print(f"{test_prompt7 = }")
    print(f"{local_decoded_generation_text7 = }")
    print(f"{test_prompt8 = }")
    print(f"{local_decoded_generation_text8 = }")
    print(f"{test_prompt9 = }")
    print(f"{local_decoded_generation_text9 = }")
    print(f"{test_prompt10 = }")
    print(f"{local_decoded_generation_text10 = }")
    print(f"{test_prompt11 = }")
    print(f"{local_decoded_generation_text11 = }")
    # Loads and tests the remote tokenizer and the remote model.
    #huggingface_username = HfApi().whoami()["name"]
    #remote_tokenizer, remote_model = load_tokenizer_and_model(os.path.join(huggingface_username, args.output_model_name))
    #remote_decoded_generation_text = test_tokenizer_and_model(remote_tokenizer, remote_model, args.test_prompt_text)

    # Checks the generated text briefly.
    #print()
    #print(f"{args.test_prompt_text = }")
    #print(f"{remote_decoded_generation_text = }")
    #print()
    #if len(remote_decoded_generation_text) <= len(args.test_prompt_text):
    #    print("Error: The generated text should not be shorter than the prompt text."
    #          " Something went wrong, so please check either the remote tokenizer or the remote model.")
    #    return


if __name__ == "__main__":
    main()
