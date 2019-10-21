import string
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt

class Reader:

    def read(self, path):
        data = pd.read_csv(path, encoding="utf-8")
        return data
class Cleaner:
    def __init__(self, sample_list):
        self.sents_list = sample_list
        self.words_list = [self.splitter(w) for w in sample_list]

    def splitter(self, sample_list):
        pos_words = sample_list.split()
        return pos_words

    def remove_punc(self):
        removed_punc = []
        table = str.maketrans('', '', string.punctuation)
        for s in self.words_list:
            removed_punc.append([w.translate(table) for w in s])
        self.words_list = removed_punc

    def lowercase(self):
        lowered = []
        for s in self.words_list:
            lowered.append([w.lower() for w in s])
        self.words_list = lowered

    def joined(self):
        sents = []
        for s in self.words_list:
            sents.append(' '.join(s))
        return sents


class drawer:
    def drawing(self, sents_list, tag_list):
        # 20 class, 20 graph
        unique_tag = ['worldnews', 'canada', 'AskReddit', 'wow', 'conspiracy',
                      'nba', 'leagueoflegends', 'soccer', 'funny', 'movies',
                      'anime', 'Overwatch', 'trees', 'GlobalOffensive', 'nfl',
                      'europe', 'Music', 'baseball', 'hockey', 'gameofthrones']
        seperate_class_list = []
        for tag in unique_tag:
            same_class_list = []
            for i in range(len(tag_list)):
                if (tag_list[i] == tag):
                    # cut by the word
                    same_class_list.append(len(sents_list[i].split(" ")))
            seperate_class_list.append(same_class_list)
        return seperate_class_list


def plot_generate():
    plt.figure("Frequency and length plot", figsize=(16, 10))
    for i in range(20):
        ax = sbn.distplot(seperate_class_list[i], kde=False, label=unique_tag[i])
    plt.legend()
    ax.set(xlabel='length of comments', ylabel='frequency')
    plt.show()


def statistic_generate():
    df = pd.DataFrame.from_records(seperate_class_list, unique_tag)
    df = df.T
    statics = df.describe()
    return statics

if __name__ == '__main__':
    data_raw = Reader().read("reddit_train.csv")
    data_train = data_raw['comments']
    data_test = data_raw['subreddits']
    # use_lemmer,use_stemmer, use_stopwords
    c = Cleaner(data_train)
    c.lowercase()
    c.remove_punc()
    data_train = c.joined()
    seperate_class_list = drawer().drawing(data_train, data_test)
    unique_tag = ['worldnews', 'canada', 'AskReddit', 'wow', 'conspiracy',
                  'nba', 'leagueoflegends', 'soccer', 'funny', 'movies',
                  'anime', 'Overwatch', 'trees', 'GlobalOffensive', 'nfl',
                  'europe', 'Music', 'baseball', 'hockey', 'gameofthrones']
    seperate_class_list = drawer().drawing(data_train, data_test)
    print(statistic_generate())
    plot_generate()