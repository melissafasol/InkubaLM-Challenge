# filename: sentiment_supplement.py

import pandas as pd
import random
import uuid

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
    df['data_source'] = source
    df['instruction'] = instruction

    cols = ['ID', 'task', 'langs', 'data_source', 'instruction', 'inputs', 'targets']
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
    df['data_source'] = source
    df['instruction'] = instruction

    cols = ['ID', 'task', 'langs', 'data_source', 'instruction', 'inputs', 'targets']
    return df[cols]
