# filename: sentiment_supplement.py

import pandas as pd
import random
import uuid
import re


def generate_swahili_supplement(n_pos=100, n_neg=100, task="sentiment", lang="swahili", source="synthetic"):
    instruction = (
        "Changanua mawazo ya matini yanayofuata na uainishe matini hayo katika mojawapo ya lebo zifuatazo. "
        "Chanya: iwapo matini yanadokeza mawazo, mtazamo na hali chanya ya kihisia. "
        "Hasi: iwapo matini yanadokeza mawazo au hisia hasi. "
        "Wastani: iwapo matini hayadokezi lugha chanya au hasi kwa njia ya moja kwa moja au isiyo ya moja kwa moja."
    )

    positive_phrases = [
        "Hongera kwa kazi nzuri!", "Nimefurahi sana kusikia hivyo.", "Hii ni habari njema sana.",
        "Asante kwa msaada wako mkubwa.", "Tunashukuru sana kwa jitihada zako.", "Hili jambo linatia moyo.",
        "Nimevutiwa na kazi yako.", "Unafanya vizuri sana.", "Mafanikio haya yanatia moyo.",
        "Ni heshima kubwa kushirikiana na wewe.", "Siku yangu imekuwa bora kwa sababu yako.",
        "Nimejifunza mengi leo.", "Ujumbe huu unatia matumaini.", "Endelea hivyo hivyo!",
        "Tunakupongeza kwa jitihada zako.", "Mkutano huo ulikuwa wa mafanikio.",
        "Huduma ilikuwa ya kiwango cha juu.", "Hili limeongeza thamani kubwa.",
        "Napenda mtazamo wako.", "Kazi yako ni bora kabisa."
    ]

    negative_phrases = [
        "Huduma ilikuwa mbaya sana.", "Nimechoshwa na tabia kama hizi.", "Hii ni aibu kubwa.",
        "Maamuzi haya hayana msingi wowote.", "Tumeshindwa kabisa kuelewa msimamo wenu.",
        "Nimepoteza imani kabisa.", "Hakuna maendeleo yanayoonekana.", "Hali inazidi kuwa mbaya.",
        "Mkutano haukuwa na tija yoyote.", "Hili jambo limeniacha na maswali mengi.",
        "Sioni faida ya kufanya hivi.", "Hali ni ya kusikitisha sana.", "Nimekata tamaa kabisa.",
        "Huduma imezorota sana.", "Tunaomba mabadiliko ya haraka.", "Sio mara ya kwanza kutokea hivi.",
        "Maelezo yako hayaridhishi.", "Uongozi umeshindwa kuonyesha dira.", "Kwa kweli hali ni mbaya.",
        "Hatujakutana na hali mbaya kama hii."
    ]

    pos_samples = [random.choice(positive_phrases) for _ in range(n_pos)]
    neg_samples = [random.choice(negative_phrases) for _ in range(n_neg)]

    samples = [{"inputs": text, "targets": "Chanya"} for text in pos_samples] + \
              [{"inputs": text, "targets": "Hasi"} for text in neg_samples]

    df = pd.DataFrame(samples)
    df['ID'] = ["ID_" + uuid.uuid4().hex[:8] + "_sentiment_dev_swahili" for _ in range(len(df))]
    df['task'] = task
    df['langs'] = lang
    df['instruction'] = instruction

    cols = ['ID', 'task', 'langs', 'instruction', 'inputs', 'targets']
    return df[cols]


def generate_hausa_supplement(n_pos=100, n_neg=100, n_neu=100, task="sentiment", lang="hausa", source="synthetic"):
    instruction = (
        "Da fatan za a gano ra'ayin da ke cikin wannan rubutu bisa ga jagorori masu zuwa: "
        "Kyakkyawa: idan rubutu na nuna kyakkyawan tunani, hali, da yanayi. "
        "Korau: idan rubutu yana nuna mummunar tunani ko yanayi. "
        "Tsaka-tsaki: idan rubutu baya nuna kyakkyawar magana ko mara kyau kai tsaye ko a kaikaice."
    )

    positive_phrases = [
        "Nagode sosai, wannan abu ya taimaka min kwarai.",
        "Lallai wannan magana tana da amfani sosai.",
        "Ina matukar jin dadin wannan aiki.",
        "Kai ne gwarzon wannan makon!",
        "Ina alfahari da kai da abin da ka aikata.",
        "Aikin nan ya burge ni matuka.",
        "Zan so ci gaba da aiki da kai.",
        "Wannan kyauta ce mai daraja.",
        "Wannan labari yana cike da farin ciki.",
        "Ina jin dadin irin kulawar da muke samu.",
    ]

    negative_phrases = [
        "Ban gamsu da wannan aikin ba ko kadan.",
        "Wannan abu ya bani takaici sosai.",
        "Ina ganin wannan lamari bai dace ba.",
        "Ka gaza cikawa alkawarin da ka dauka.",
        "Halin da ake ciki yana kara tabarbarewa.",
        "Ina fushi da yadda aka gudanar da wannan shiri.",
        "Wannan magana bata da tushe ko makama.",
        "Ba zan iya yarda da hakan ba.",
        "Na damu matuka da wannan sakamako.",
        "Wannan ba shine abin da aka alkawarta ba.",
    ]

    neutral_phrases = [
        "Yau rana ce mai kyau a gari.",
        "Ina zaune a gida ina kallon talabijin.",
        "Za mu je kasuwa gobe da safe.",
        "Ina da aiki da yawa a yau.",
        "Mutumin nan yana da shekaru talatin da biyu.",
        "Gobe za mu fara sabon zangon karatu.",
        "Na ci abinci da rana.",
        "Za mu gana da su bayan sallar magariba.",
        "Ina jiran zuwan motar haya.",
        "Na karanta wata kasida mai kayatarwa jiya.",
    ]

    def make_samples(phrases, label, n):
        return [{"inputs": random.choice(phrases), "targets": label} for _ in range(n)]

    data = (
        make_samples(positive_phrases, "Kyakkyawa", n_pos) +
        make_samples(negative_phrases, "Korau", n_neg) +
        make_samples(neutral_phrases, "Tsaka-tsaki", n_neu)
    )

    df = pd.DataFrame(data)
    df['ID'] = ["ID_" + uuid.uuid4().hex[:8] + "_sentiment_dev_hausa" for _ in range(len(df))]
    df['task'] = task
    df['langs'] = lang
    df['instruction'] = instruction

    cols = ['ID', 'task', 'langs', 'instruction', 'inputs', 'targets']
    return df[cols]


def generate_afrixnli_swahili(n_entail=100, n_neutral=100, n_contra=100, source="synthetic"):
    instruction = "Is the following question True, False or Neither?"

    premise_pairs = {
        0: [  # Entailment (True)
              "Alikula chakula chake cha mchana.",
        "Alifanya mazoezi kila siku.",
        "Alikuwa na njaa kabla ya kula.",
        "Alikamilisha kazi kabla ya muda kuisha.",
        "Aliamka mapema leo asubuhi.",
        "Wanafunzi walihudhuria somo la leo.",
        "Yeye ni daktari wa hospitali kuu.",
        "Walipata matokeo mazuri ya mtihani.",
        "Walisafiri kwenda Mombasa wikendi hii.",
        "Mtoto alicheza nje mchana kutwa.",
        "Aliandika ripoti kwa bidii.",
        "Walimaliza kusafisha nyumba kabla ya wageni kuwasili.",
        "Alifunga duka saa mbili usiku.",
        "Yeye ni mwanafunzi wa mwaka wa pili.",
        "Aliwasiliana na mwalimu kuhusu mradi.",
        "Walipanda miti kwenye uwanja wa shule.",
        "Aliendesha gari hadi kazini.",
        "Walienda sokoni kununua matunda.",
        "Alioga kabla ya kwenda shuleni.",
        "Alizungumza na mama yake kwa simu.",
        "Alihudhuria mkutano wote.",
        "Alifagia ofisi kila asubuhi.",
        "Aliandika barua ya kuomba kazi.",
        "Walifurahia sinema waliyoitazama.",
        "Alihudumu katika jeshi kwa miaka kumi."],
        1: [  # Neutral (Neither)
            "Ana gari kubwa aina ya Toyota.",
        "Alivaa fulana nyekundu jana.",
        "Saa yake mpya ni ya bei ghali.",
        "Anaishi karibu na mto.",
        "Anapenda kusoma vitabu vya historia.",
        "Chakula cha jioni kilikuwa kitamu.",
        "Alikuwa akisoma gazeti wakati simu ililia.",
        "Alitembea kwa muda mrefu leo.",
        "Walienda kwenye duka la vitabu.",
        "Anapendelea kahawa kuliko chai.",
        "Alikaa kimya muda mrefu.",
        "Wana mpango wa kujenga nyumba mwaka huu.",
        "Aliwahi kuwa na mbwa wa kufuga.",
        "Ana wazo kuhusu biashara mpya.",
        "Alisafiri kwa ndege mara ya mwisho mwezi uliopita.",
        "Kuna miti mingi kwenye shamba lake.",
        "Aliangalia tamasha la muziki jana.",
        "Ana ndoto ya kuwa mwandishi maarufu.",
        "Anapenda muziki wa bongo fleva.",
        "Walitembelea jumba la makumbusho.",
        "Alinunua kalamu mpya ya wino wa bluu.",
        "Hawakuwahi kula chakula cha India.",
        "Aliwahi kufika Nairobi mara moja.",
        "Alikuwa akisubiri matokeo ya kura.",
        "Anafikiria kuhamia mjini mwaka huu."
            ],
        2: [  # Contradiction (False)
            "Hakufika kazini leo.",
    "Alikataa kula chakula alichoandaliwa.",
    "Alisema hajawahi kuona bahari.",
    "Alisahau namba yake ya siri.",
    "Alikataa kuhudhuria mkutano.",
    "Hajawahi kuendesha gari maishani.",
    "Aliharibu kazi yote aliyofanya jana.",
    "Anachukia muziki kabisa.",
    "Hajawahi kutoka nje ya nchi.",
    "Hajui kuandika wala kusoma.",
    "Alisema hawezi kuogelea.",
    "Aliwahi kusema hapendi chai.",
    "Hapendi wanyama wa kufugwa.",
    "Anasema hakuwahi kukutana na mwalimu huyo.",
    "Alikataa kutoa msaada wowote.",
    "Alikataa kupokea zawadi aliyopewa.",
    "Hakuwahi kupanda ndege.",
    "Alikana kuwa aliandika barua hiyo.",
    "Alisema haoni umuhimu wa shule.",
    "Hajawahi kufika kazini mapema.",
    "Aliharibu kompyuta kwa makusudi.",
    "Alisema siyo kazi yake kufagia.",
    "Aliwahi kusema hafurahii kazi hiyo.",
    "Alikataa kabisa kushiriki katika shughuli hiyo.",
    "Alikana kumjua mtu huyo."
        ]
    }

    def make_samples(label_id, n):
        return [{
            "inputs": random.choice(premise_pairs[label_id]),
            "targets": label_id
        } for _ in range(n)]

    samples = (
        make_samples(0, n_entail) +
        make_samples(1, n_neutral) +
        make_samples(2, n_contra)
    )

    df = pd.DataFrame(samples)
    df['ID'] = ["ID_" + uuid.uuid4().hex[:8] + "_dev_afrixnli_swa" for _ in range(len(df))]
    df['task'] = "afrixnli"
    df['langs'] = "swa"
    df['instruction'] = instruction

    cols = ['ID', 'langs', 'instruction', 'inputs', 'targets', 'task']
    return df[cols]


def generate_afrixnli_hausa(n_entail=100, n_neutral=100, n_contra=100, source="synthetic"):
    instruction = "Is the following question True, False or Neither?"

    premise_pairs = {
        0: [  # Entailment (True)
            "Ya ci abincinsa kafin tafiya.",
    "Sun gama aikin kafin rana ta fadi.",
    "Yana zuwa aiki kowace rana.",
    "Ya rubuta wasikar godiya.",
    "Yana karanta littafi a dakin karatu.",
    "Ta share dakin sosai.",
    "Yana da lafiya sosai.",
    "Ta shirya abinci kafin maigida ya dawo.",
    "Suka tafi kasuwa da safe.",
    "Yana amfani da mota zuwa makaranta.",
    "Ta kammala aikin kafin lokaci ya cika.",
    "Ya yi wanka kafin ya fita.",
    "Yana kwana a gidan iyayensa.",
    "Sun je asibiti don duba marar lafiya.",
    "Ta gama karatu da dare.",
    "Yana da kwarewa a harkar lissafi.",
    "Ya shirya jawabinsa kafin taron.",
    "Ya kira mahaifiyarsa jiya.",
    "Ta halarci taron gaba daya.",
    "Sun dauki hoto tare da shugaban makaranta."
        ],
        1: [  # Neutral (Neither)
             "Yana da jakar makaranta mai launin ja.",
    "Ta fi son shinkafa da miya.",
    "Yana jin dadin kallon fina-finai.",
    "Ta dafa tuwo da miyar kuka.",
    "Suna shan lemo a shagon mai gida.",
    "Yana amfani da kwamfutar tafi-da-gidanka.",
    "Ta kan karanta jarida da safe.",
    "Yana zaune kusa da kasuwa.",
    "Sun sayi sabuwar firiji.",
    "Ya fi son kifi fiye da nama.",
    "Yana jin dadin yawo da yamma.",
    "Yana fatan zuwa kasar waje.",
    "Ta saba da tafiya da keke.",
    "Suna shan ruwan sanyi sosai.",
    "Ya fi son karatun lissafi.",
    "Yana so ya koyi sana’a.",
    "Sun je gidan makwabta jiya.",
    "Yana son rakumi sosai.",
    "Ya shirya biki mai kyau.",
    "Yana son tafiye-tafiye na nishaɗi."
        ],
        2: [  # Contradiction (False)
            "Bai je aiki ba yau.",
    "Ya ki cin abinci da daddare.",
    "Ya ce ba zai taba zuwa makaranta ba.",
    "Ta ki karanta littafin da aka bata.",
    "Ba ya jin dadin tafiya ko kadan.",
    "Ya taba cewa baya son ruwan sanyi.",
    "Ta ce bata san girki ba.",
    "Ya ce ba zai taba yarda da hakan ba.",
    "Ya fadi cewa baya jin dadin fim.",
    "Ya taba cewa bai taba zuwa kasuwa ba.",
    "Ba ya son komai da ya shafi karatu.",
    "Ya ce bai taba ganin wannan mutum ba.",
    "Ta fadi cewa bata taba cin shinkafa ba.",
    "Ya ce baya da kowane irin buri.",
    "Ya taba cewa bashi da abokai.",
    "Ta fadi cewa bata taba ganin kifi ba.",
    "Ya ce ba ya amfani da waya.",
    "Ta taba cewa bata jin dadin tafiya.",
    "Ya fadi cewa baya da sha'awar zama likita.",
    "Bai taba zuwa gidansu ba."
        ]
    }

    def make_samples(label_id, n):
        return [{
            "inputs": random.choice(premise_pairs[label_id]),
            "targets": label_id
        } for _ in range(n)]

    samples = (
        make_samples(0, n_entail) +
        make_samples(1, n_neutral) +
        make_samples(2, n_contra)
    )

    df = pd.DataFrame(samples)
    df['ID'] = ["ID_" + uuid.uuid4().hex[:8] + "_dev_afrixnli_hau" for _ in range(len(df))]
    df['task'] = "afrixnli"
    df['langs'] = "hau"
    df['instruction'] = instruction

    cols = ['ID', 'langs', 'instruction', 'inputs', 'targets', 'task']
    return df[cols]


# === Shared Utilities ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r'@[\w_]+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    emoji_pattern = re.compile(
        "[" u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF"
        u"\U00002700-\U000027BF" u"\U000024C2-\U0001F251" "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r"[^\w\s.,!?'-]", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def has_repeated_punctuation(text, threshold=3):
    return bool(re.search(rf'([^\w\s])\1{{{threshold - 1},}}', text))

def has_repeated_words(text, threshold=3):
    words = text.split()
    count = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            count += 1
            if count >= threshold:
                return True
        else:
            count = 1
    return False

def update_instruction(row):
    task, lang = row["task"], row["langs"]
    
    if task == "sentiment":
        if lang == "swa":
            return "Aina ya hisia: Chanya, Hasi, au Wastani."
        elif lang == "hausa":
            return "Nuna yanayi: Kyakkyawa, Korau, ko Tsaka-tsaki."

    elif task == "mt":
        if lang == "swa":
            return "Tafsiri kwa Kiswahili."
        elif lang == "hausa":
            return "Fassara zuwa Hausa."

    elif task == "afrixnli":
        if lang == "swa":
            return "Je hoja inathibitisha (0), haina uhusiano (1), au inapingana (2) na kauli?"
        elif lang == "hau":
            return "Shin bayanin na goyon baya (0), ba shi da alaƙa (1), ko yana ƙaryata (2) ƙalmar?"

    return row["instruction"]  # fallback if no match

# === Sentiment Preprocessing ===
def preprocess_sentiment_dataframe(df):
    df = df.copy()
    df['inputs'] = df['inputs'].astype(str).apply(clean_text)

    # Filter out problematic rows
    to_remove = df['inputs'].apply(has_repeated_punctuation) | df['inputs'].apply(has_repeated_words)
    removed_df = df[to_remove].copy()
    cleaned_df = df[~to_remove].copy()

    # Normalize labels
    label_map = {
        'kyakkyawa': 'positive', 'tsaka-tsaki': 'neutral', 'wastani': 'neutral',
        'korau': 'negative', 'positive': 'positive', 'neutral': 'neutral', 'negative': 'negative'
    }
    cleaned_df['targets'] = cleaned_df['targets'].str.lower().map(label_map)
    cleaned_df = cleaned_df.dropna(subset=['targets'])

    # Update instruction field
    cleaned_df["instruction"] = cleaned_df.apply(update_instruction, axis=1)

    return cleaned_df, removed_df

# === Machine Translation Preprocessing ===
def preprocess_mt_dataframe(df):
    df = df.copy()

    df['inputs'] = df['inputs'].str.strip()
    df['targets'] = df['targets'].str.strip()
    df['instruction'] = df['instruction'].str.strip()

    df['inputs'] = df['inputs'].str.replace(r"&gt;", ">", regex=True)
    df['targets'] = df['targets'].str.replace(r"&gt;", ">", regex=True)

    df = df.dropna(subset=['inputs', 'targets'])
    df = df[df['inputs'].str.len() > 3]
    df = df[df['targets'].str.len() > 3]

    df['inputs'] = df['inputs'].apply(clean_text)
    df['targets'] = df['targets'].apply(clean_text)

    # Update instruction field
    df["instruction"] = df.apply(update_instruction, axis=1)

    df['text_input'] = df['instruction'] + "\n" + df['inputs']
    return df[['ID', 'langs', 'instruction', 'inputs', 'targets', 'text_input']]


def preprocess_nli_dataframe(df):
    df = df.copy()

    # Clean + normalize text
    df['inputs'] = df['inputs'].astype(str).str.strip().str.lower()
    df['instruction'] = df['instruction'].astype(str).str.strip()

    # Assign numeric, task-specific instruction
    def assign_classification_instruction(row):
        if row["langs"] == "swa":
            return "Chagua jibu sahihi: 0 (Chanya), 1 (Wastani), 2 (Hasi)."
        elif row["langs"] == "hau":
            return "Zaɓi: 0 (Kyakkyawa), 1 (Tsaka-tsaki), 2 (Korau)."
        else:
            return "Chagua daraja sahihi: 0, 1, au 2."

    df["instruction"] = df.apply(assign_classification_instruction, axis=1)

    # Build simple prompt
    df['text_input'] = df['instruction'] + "\n" + df['inputs']

    return df[['ID', 'langs', 'instruction', 'inputs', 'text_input', 'targets']]
