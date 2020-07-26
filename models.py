#-*- coding:utf-8 -*-
import warnings
import platform
import os
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing as pp
import scipy.sparse as spr
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from selenium import webdriver

class HYBRID_CBF_ICBF:
    def __init__(self, train_df, test_df):

        warnings.filterwarnings("ignore")

        self.train = train_df
        self.test = test_df
        self.stop_words = self.crwal_stopwords()

    def execute_recommendation(self):

        plylst_train, plylst_test, train_songs_A, train_tags_A, song_dict, tag_dict, plylst_nid_id, song_sid_id, tag_tid_id = self.clean_b4_train()
        result = self.predict(plylst_train,plylst_test,train_songs_A,train_tags_A, song_dict, tag_dict, plylst_nid_id, song_sid_id, tag_tid_id)

        return result

    def clean_b4_train(self):
        train, test = self.train, self.test

        train['istrain'] = 1
        test['istrain'] = 0

        n_train = len(train)
        n_test = len(test)

        # train + test
        plylst = pd.concat([train, test], ignore_index=True)

        # playlist id
        plylst["nid"] = range(n_train + n_test)

        # id <-> nid
        plylst_id_nid = dict(zip(plylst["id"], plylst["nid"]))
        plylst_nid_id = dict(zip(plylst["nid"], plylst["id"]))

        plylst_tag = plylst['tags']
        tag_counter = Counter([tg for tgs in plylst_tag for tg in tgs])
        tag_dict = {x: tag_counter[x] for x in tag_counter}

        tag_id_tid = dict()
        tag_tid_id = dict()
        for i, t in enumerate(tag_dict):
            # print(t)
            tag_id_tid[t] = i
            tag_tid_id[i] = t

        n_tags = len(tag_dict)

        plylst_song = plylst['songs']
        song_counter = Counter([sg for sgs in plylst_song for sg in sgs])
        song_dict = {x: song_counter[x] for x in song_counter}

        song_id_sid = dict()
        song_sid_id = dict()
        for i, t in enumerate(song_dict):
            song_id_sid[t] = i
            song_sid_id[i] = t

        n_songs = len(song_dict)

        plylst['songs_id'] = plylst['songs'].map(
            lambda x: [song_id_sid.get(s) for s in x if song_id_sid.get(s) != None])
        plylst['tags_id'] = plylst['tags'].map(lambda x: [tag_id_tid.get(t) for t in x if tag_id_tid.get(t) != None])

        # plylst_use = plylst[['istrain', 'nid', 'updt_date', 'songs_id', 'tags_id']]
        plylst_use = plylst.copy()
        plylst_use.loc[:, 'num_songs'] = plylst_use['songs_id'].map(len)
        plylst_use.loc[:, 'num_tags'] = plylst_use['tags_id'].map(len)
        plylst_use = plylst_use.set_index('nid')

        plylst_train = plylst_use.iloc[:n_train, :]
        plylst_test = plylst_use.iloc[n_train:, :]

        row = np.repeat(range(n_train), plylst_train['num_songs'])
        col = [song for songs in plylst_train['songs_id'] for song in songs]
        dat = np.repeat(1, plylst_train['num_songs'].sum())
        train_songs_A = spr.csr_matrix((dat, (row, col)), shape=(n_train, n_songs))

        row = np.repeat(range(n_train), plylst_train['num_tags'])
        col = [tag for tags in plylst_train['tags_id'] for tag in tags]
        dat = np.repeat(1, plylst_train['num_tags'].sum())
        train_tags_A = spr.csr_matrix((dat, (row, col)), shape=(n_train, n_tags))

        return plylst_train, plylst_test, train_songs_A, train_tags_A, song_dict, tag_dict, plylst_nid_id, song_sid_id, tag_tid_id

    def predict(self, train_df, test_df, train_songs_mat, train_tags_mat, song_dic, tag_dic, plylst_nid_id_dic, song_sid_id_dic, tag_tid_id_dic):

        plylst_train = train_df
        plylst_test = test_df
        train_songs_A = train_songs_mat
        train_tags_A = train_tags_mat
        song_dict = song_dic
        tag_dict = tag_dic
        plylst_nid_id = plylst_nid_id_dic
        song_sid_id = song_sid_id_dic
        tag_tid_id = tag_tid_id_dic

        ## 1. 노래
        ###  1.1. songs 값이 있는경우
        ###  1.2. songs값이 없는 경우 (plylst_title 이 있음)
        ###  1.3. 둘 다 아닌 경우

        plylst_test_is_songs = plylst_test.loc[plylst_test.num_songs > 0]
        plylst_test_no_songs = plylst_test.loc[
            np.logical_and(plylst_test.num_songs == 0, plylst_test.plylst_title != '')]
        plylst_test_no_songs_title = plylst_test.loc[
            np.logical_and(plylst_test.num_songs == 0, plylst_test.plylst_title == '')]

        ## 2. 태그
        ###  2.1. tags 값이 있는 경우
        ###  2.2. tags 값이 없는 경우 (plylst_title 이 있음)
        ###  2.3. 둘 다 아닌 경우

        plylst_test_is_tags = plylst_test.loc[plylst_test.num_tags > 0]
        plylst_test_no_tags = plylst_test.loc[np.logical_and(plylst_test.num_tags == 0, plylst_test.plylst_title != '')]
        plylst_test_no_tags_title = plylst_test.loc[
            np.logical_and(plylst_test.num_tags == 0, plylst_test.plylst_title == '')]

        song1 = self.icbf(plylst_train, plylst_test, train_songs_A, train_tags_A, song_dict, tag_dict, plylst_nid_id, song_sid_id, tag_tid_id, p_ids=plylst_test_is_songs.index, is_song=True)
        tag1 = self.icbf(plylst_train, plylst_test, train_songs_A, train_tags_A, song_dict, tag_dict, plylst_nid_id, song_sid_id,
             tag_tid_id, p_ids=plylst_test_is_tags.index, is_song=False)


        song2 = self.cbf(plylst_train, plylst_test, train_songs_A, train_tags_A, song_dict, tag_dict, plylst_nid_id, song_sid_id, tag_tid_id, p_ids=plylst_test_no_songs.index, is_song=True)
        tag2 = self.cbf(plylst_train, plylst_test, train_songs_A, train_tags_A, song_dict, tag_dict, plylst_nid_id, song_sid_id,
             tag_tid_id, p_ids=plylst_test_no_tags.index, is_song=False)

        song3 = self.reminder(plylst_train, plylst_test, train_songs_A, train_tags_A, song_dict, tag_dict, plylst_nid_id, song_sid_id, tag_tid_id, p_ids=plylst_test_no_songs_title.index, is_song=True)
        tag3 = self.reminder(plylst_train, plylst_test, train_songs_A, train_tags_A, song_dict, tag_dict, plylst_nid_id, song_sid_id,
             tag_tid_id, p_ids=plylst_test_no_tags_title.index, is_song=False)

        song_dicts = [song1, song2, song3]
        total_song_dict = dict()

        for song_dict in song_dicts:
            for key, value in song_dict.items():
                total_song_dict[key] = value

        tag_dicts = [tag1, tag2, tag3]
        total_tag_dict = dict()

        for tag_dict in tag_dicts:
            for key, value in tag_dict.items():
                total_tag_dict[key] = value

        test_plylst_ids = list(total_song_dict.keys())

        result = list()

        for id in test_plylst_ids:
            result.append({
                "id": id,
                "songs": total_song_dict[id],
                "tags": total_tag_dict[id]
            })

        return result


    def icbf(self,train_df, test_df, train_songs_mat, train_tags_mat, song_dic, tag_dic, plylst_nid_id_dic, song_sid_id_dic, tag_tid_id_dic, p_ids , is_song=True):

        plylst_train = train_df
        plylst_test = test_df

        train_songs_A = train_songs_mat
        train_tags_A = train_tags_mat
        song_dict = song_dic
        tag_dict = tag_dic
        plylst_nid_id = plylst_nid_id_dic
        song_sid_id = song_sid_id_dic
        tag_tid_id = tag_tid_id_dic
        pids = p_ids

        n_songs = len(song_dict)
        n_tags = len(tag_dict)

        res = dict()

        if is_song == True:
            target = "songs"
        else:
            target = "tags"
        print(f"icbf of {target} is processing...")

        if is_song == True:

            train_songs_A_T = train_songs_A.transpose()

            desc_occ_song_pid = sorted(song_dict.items(), key=lambda kv: kv[1], reverse=True)
            top_occ_song_id = [song_id for song_id, value in desc_occ_song_pid[:100]]

            for pid in tqdm(pids):
                p = np.zeros((n_songs, 1))
                # 하나의 테스트 플레이리스트 기준 있는 노래에 1을 표기
                p[plylst_test.loc[pid, 'songs_id']] = 1
                # user X item 내적 (item X 1) => user X 1 [115071, 1] => [115071, ] => 하나의 플레이리스트 기준 다른 플레이리스트들과의 상관관계 값?
                val = train_songs_A.dot(p).reshape(-1)

                # test의 song 값 test의 tag값
                songs_already = plylst_test.loc[pid, "songs_id"]

                # Item X User 매트릭스와 하나의 플레이리스트를 기준으로 유사한 점수를 나타내는 매트릭스와 내적 => 노래들의 값
                cand_song = train_songs_A_T.dot(val)  # (638336, 1)
                cand_song_idx = cand_song.reshape(-1).argsort()[-150:][::-1]  # 상위 150개 노래 도출
                cand_song_pnt = np.sort(cand_song.reshape(-1))[::-1][:150]

                # 이미 노래가 있는 값을 제거하고 상위 100개
                cand_song_idx = cand_song_idx[np.isin(cand_song_idx, songs_already) == False][:100]
                # 기존 ID 값 converting 하여 결과에 담기
                rec_song_idx = [song_sid_id[i] for i in cand_song_idx]

                if len(rec_song_idx) != 100:
                    lack_num = 100 - len(rec_song_idx)
                    # target_songs = list(set(top_occ_song_id).difference(set(rec_song_idx)))
                    target_songs = [song_id for song_id in top_occ_song_id if song_id not in rec_song_idx]
                    song_reminder = target_songs[:lack_num]
                    rec_song_idx = rec_song_idx + song_reminder

                res[plylst_nid_id[pid]] = rec_song_idx

        else:

            train_tags_A_T = train_tags_A.transpose()

            desc_occ_tag_pid = sorted(tag_dict.items(), key=lambda kv: kv[1], reverse=True)
            top_occ_tag_id = [tag_id for tag_id, value in desc_occ_tag_pid[:10]]

            for pid in tqdm(pids):
                # (30197,1)
                v = np.zeros((n_tags, 1))
                v[plylst_test.loc[pid, 'tags_id']] = 1

                # (115071,30197) (30197,1) => (115071,1)
                val = train_tags_A.dot(v).reshape(-1)

                tags_already = plylst_test.loc[pid, "tags_id"]

                # (30197,115071) (115071,1) => (30197,1)
                cand_tag = train_tags_A_T.dot(val)
                cand_tag_idx = cand_tag.reshape(-1).argsort()[-15:][::-1]
                cand_tag_pnt = np.sort(cand_tag.reshape(-1))[::-1][:15]

                cand_tag_idx = cand_tag_idx[np.isin(cand_tag_idx, tags_already) == False][:10]
                rec_tag_idx = [tag_tid_id[i] for i in cand_tag_idx]

                if len(rec_tag_idx) != 10:
                    lack_num = 10 - len(rec_tag_idx)
                    # target_tags = list(set(top_occ_tag_id).difference(set(rec_tag_idx)))
                    target_tags = [tag_id for tag_id in top_occ_tag_id if tag_id not in rec_tag_idx]
                    tag_reminder = target_tags[:lack_num]
                    # tag_reminder = top_occ_tag_id[:lack_num]
                    rec_tag_idx = rec_tag_idx + tag_reminder

                res[plylst_nid_id[pid]] = rec_tag_idx

        return res


    def cbf(self, train_df, test_df, train_songs_mat, train_tags_mat, song_dic, tag_dic, plylst_nid_id_dic,
             song_sid_id_dic, tag_tid_id_dic, p_ids, is_song=True):

        plylst_train = train_df
        plylst_test = test_df

        train_songs_A = train_songs_mat
        train_tags_A = train_tags_mat
        song_dict = song_dic
        tag_dict = tag_dic
        plylst_nid_id = plylst_nid_id_dic
        song_sid_id = song_sid_id_dic
        tag_tid_id = tag_tid_id_dic
        pids = p_ids

        n_songs = len(song_dict)
        n_tags = len(tag_dict)

        res = dict()

        if is_song == True:
            target = "songs"
        else:
            target = "tags"
        print(f"cbf of {target} is processing...")

        stop_words = self.stop_words

        if is_song == True:

            plylst_test_no_songs = plylst_test.loc[pids]
            data = pd.concat([plylst_train, plylst_test_no_songs], axis=0)

            n_train = plylst_train.shape[0]
            n_all = data.shape[0]

            tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=3, stop_words=stop_words)
            data_tfidf_title = tf.fit_transform(data.plylst_title)

            user_cosine = self.cosine_similarities(data_tfidf_title.transpose())

            new_cid = list(range(n_all))
            origin_nid = list(data.index)

            plylst_cid_nid = dict(zip(new_cid, origin_nid))
            plylst_nid_cid = dict(zip(origin_nid, new_cid))

            for t_pid in tqdm(range(n_train, n_all)):
                top_n_plylsts = np.argsort(user_cosine[t_pid].todense().A1)[:-1][1:11]
                top_n_plylsts = [plylst_cid_nid[top_plylst] for top_plylst in top_n_plylsts]
                top_n_plylsts_dict = OrderedDict()

                for plylst in top_n_plylsts:
                    target_songs = data.loc[data.index == plylst].songs
                    target_songs_rank = {rank + 1: song for target_song in target_songs for rank, song in
                                         enumerate(target_song)}
                    top_n_plylsts_dict[plylst] = target_songs_rank

                ranking_dict = dict()

                for idx, (pid, rank_dict) in enumerate(top_n_plylsts_dict.items()):
                    if idx == 0:
                        weight = 1
                    else:
                        weight = 1 + (2 ** idx / 2 ** 9)
                    for rank, song_id in rank_dict.items():
                        if ranking_dict.get(song_id) is None:
                            ranking_dict[song_id] = {"occ_count": 1, "rank": rank * weight}
                        else:
                            ranking_dict[song_id]["occ_count"] += 1
                            ranking_dict[song_id]["rank"] += rank * weight

                mean_rank_dict = dict()
                for song_id, rank_dict in ranking_dict.items():
                    mean_rank = rank_dict["rank"] / rank_dict["occ_count"]
                    mean_rank_dict[song_id] = mean_rank

                sorted_mean_rank = sorted(mean_rank_dict.items(), key=lambda kv: kv[1], reverse=False)
                cbf_song_list = list(map(lambda x: x[0], sorted_mean_rank[:100]))
                res[plylst_nid_id[plylst_cid_nid[t_pid]]] = cbf_song_list

        else:
            plylst_test_no_tags = plylst_test.loc[pids]
            data = pd.concat([plylst_train, plylst_test_no_tags], axis=0)

            n_train = plylst_train.shape[0]
            n_all = data.shape[0]

            tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=3, stop_words=stop_words)
            data_tfidf_title = tf.fit_transform(data.plylst_title)

            user_cosine = self.cosine_similarities(data_tfidf_title.transpose())

            new_cid = list(range(n_all))
            origin_nid = list(data.index)

            plylst_cid_nid = dict(zip(new_cid, origin_nid))
            plylst_nid_cid = dict(zip(origin_nid, new_cid))

            # cbf_result_dict = dict()
            for t_pid in tqdm(range(n_train, n_all)):
                top_n_plylsts = np.argsort(user_cosine[t_pid].todense().A1)[:-1][1:11]
                top_n_plylsts = [plylst_cid_nid[top_plylst] for top_plylst in top_n_plylsts]
                top_n_plylsts_dict = OrderedDict()

                for plylst in top_n_plylsts:
                    target_tags = data.loc[data.index == plylst].tags
                    target_tags_rank = {rank + 1: tag for target_tag in target_tags for rank, tag in
                                        enumerate(target_tag)}
                    top_n_plylsts_dict[plylst] = target_tags_rank

                ranking_dict = dict()

                for idx, (pid, rank_dict) in enumerate(top_n_plylsts_dict.items()):
                    if idx == 0:
                        weight = 1
                    else:
                        weight = 1 + (2 ** idx / 2 ** 9)
                    for rank, tag_id in rank_dict.items():
                        if ranking_dict.get(tag_id) is None:
                            ranking_dict[tag_id] = {"occ_count": 1, "rank": rank * weight}
                        else:
                            ranking_dict[tag_id]["occ_count"] += 1
                            ranking_dict[tag_id]["rank"] += rank * weight

                mean_rank_dict = dict()
                for tag_id, rank_dict in ranking_dict.items():
                    mean_rank = rank_dict["rank"] / rank_dict["occ_count"]
                    mean_rank_dict[tag_id] = mean_rank

                sorted_mean_rank = sorted(mean_rank_dict.items(), key=lambda kv: kv[1], reverse=False)
                cbf_song_list = list(map(lambda x: x[0], sorted_mean_rank[:10]))
                res[plylst_nid_id[plylst_cid_nid[t_pid]]] = cbf_song_list

        return res


    def crwal_stopwords(self):
        os_env= platform.system()

        if os_env == 'Linux':
            self.web_driver = os.path.abspath(os.path.dirname(os.path.abspath('__file__'))) + '/webdriver/chromedriver'
        elif os_env == 'Windows':
            self.web_driver = os.path.abspath(os.path.dirname(os.path.abspath('__file__'))) + '\webdriver\chromedriver.exe'

        driver = webdriver.Chrome(self.web_driver)

        driver.implicitly_wait(3)
        url = 'https://www.ranks.nl/stopwords/korean'
        driver.get(url)
        element = driver.find_element_by_id('article178ebefbfb1b165454ec9f168f545239')
        stop_words_lists = element.find_elements_by_css_selector("td")

        kr_stop_words = list()

        for stop_words_list in stop_words_lists:
            text_list = stop_words_list.text
            text_list = text_list.split("\n")
            kr_stop_words += text_list

        kr_stop_words = list(set(kr_stop_words))

        nltk.download('stopwords')
        en_stop_words = stopwords.words("english")

        stop_words = kr_stop_words + en_stop_words

        return stop_words

    def cosine_similarities(self, mat):
        col_normed_mat = pp.normalize(mat.tocsc(), axis=0)
        return col_normed_mat.T * col_normed_mat

    def reminder(self, train_df, test_df, train_songs_mat, train_tags_mat, song_dic, tag_dic, plylst_nid_id_dic,
             song_sid_id_dic, tag_tid_id_dic, p_ids, is_song=True):

        plylst_train = train_df
        plylst_test = test_df

        train_songs_A = train_songs_mat
        train_tags_A = train_tags_mat

        train_songs_A_T = train_songs_A.transpose()
        train_tags_A_T = train_tags_A.transpose()

        song_dict = song_dic
        tag_dict = tag_dic
        plylst_nid_id = plylst_nid_id_dic
        song_sid_id = song_sid_id_dic
        tag_tid_id = tag_tid_id_dic
        pids = p_ids

        n_songs = len(song_dict)
        n_tags = len(tag_dict)

        res = dict()

        if is_song == True:
            target = "songs"
        else:
            target = "tags"
        print(f"reminder of {target} is processing...")

        if is_song == True:
            desc_occ_song_pid = sorted(song_dict.items(), key=lambda kv: kv[1], reverse=True)
            top_occ_song_id = [song_id for song_id, value in desc_occ_song_pid[:100]]

            for pid in tqdm(pids):
                res[plylst_nid_id[pid]] = top_occ_song_id

        else:
            desc_occ_tag_pid = sorted(tag_dict.items(), key=lambda kv: kv[1], reverse=True)
            top_occ_tag_id = [tag_id for tag_id, value in desc_occ_tag_pid[:10]]

            for pid in tqdm(pids):
                p = np.zeros((n_songs, 1))
                # 하나의 테스트 플레이리스트 기준 있는 노래에 1을 표기
                p[plylst_test.loc[pid, 'songs_id']] = 1
                # user X item 내적 (item X 1) => user X 1 [115071, 1] => [115071, ] => 하나의 플레이리스트 기준 다른 플레이리스트들과의 상관관계 값?
                val = train_songs_A.dot(p).reshape(-1)

                tags_already = plylst_test.loc[pid, "tags_id"]

                # (30197,115071) (115071,1) => (30197,1)
                cand_tag = train_tags_A_T.dot(val)
                cand_tag_idx = cand_tag.reshape(-1).argsort()[-15:][::-1]
                cand_tag_pnt = np.sort(cand_tag.reshape(-1))[::-1][:15]

                cand_tag_idx = cand_tag_idx[np.isin(cand_tag_idx, tags_already) == False][:10]
                rec_tag_idx = [tag_tid_id[i] for i in cand_tag_idx]

                if len(rec_tag_idx) != 10:
                    lack_num = 10 - len(rec_tag_idx)
                    # target_tags = list(set(top_occ_tag_id).difference(set(rec_tag_idx)))
                    target_tags = [tag_id for tag_id in top_occ_tag_id if tag_id not in rec_tag_idx]
                    tag_reminder = target_tags[:lack_num]
                    # tag_reminder = top_occ_tag_id[:lack_num]
                    rec_tag_idx = rec_tag_idx + tag_reminder

                res[plylst_nid_id[pid]] = rec_tag_idx

        return res


class ICBF_OCC:
    def __init__(self, train_df, test_df):
        self.train = train_df
        self.test = test_df

    def execute_recommendation(self):

        plylst_train, plylst_test, train_songs_A, train_tags_A, song_dict, tag_dict, plylst_nid_id, song_sid_id, tag_tid_id = self.clean_b4_train()
        result = self.predict(plylst_train,plylst_test,train_songs_A,train_tags_A, song_dict, tag_dict, plylst_nid_id, song_sid_id, tag_tid_id)

        return result

    def clean_b4_train(self):
        train, test = self.train, self.test

        train['istrain'] = 1
        test['istrain'] = 0

        n_train = len(train)
        n_test = len(test)

        # train + test
        plylst = pd.concat([train, test], ignore_index=True)

        # playlist id
        plylst["nid"] = range(n_train + n_test)

        # id <-> nid
        plylst_id_nid = dict(zip(plylst["id"], plylst["nid"]))
        plylst_nid_id = dict(zip(plylst["nid"], plylst["id"]))

        plylst_tag = plylst['tags']
        tag_counter = Counter([tg for tgs in plylst_tag for tg in tgs])
        tag_dict = {x: tag_counter[x] for x in tag_counter}

        tag_id_tid = dict()
        tag_tid_id = dict()
        for i, t in enumerate(tag_dict):
            # print(t)
            tag_id_tid[t] = i
            tag_tid_id[i] = t

        n_tags = len(tag_dict)

        plylst_song = plylst['songs']
        song_counter = Counter([sg for sgs in plylst_song for sg in sgs])
        song_dict = {x: song_counter[x] for x in song_counter}

        song_id_sid = dict()
        song_sid_id = dict()
        for i, t in enumerate(song_dict):
            song_id_sid[t] = i
            song_sid_id[i] = t

        n_songs = len(song_dict)

        plylst['songs_id'] = plylst['songs'].map(
            lambda x: [song_id_sid.get(s) for s in x if song_id_sid.get(s) != None])
        plylst['tags_id'] = plylst['tags'].map(lambda x: [tag_id_tid.get(t) for t in x if tag_id_tid.get(t) != None])

        plylst_use = plylst[['istrain', 'nid', 'updt_date', 'songs_id', 'tags_id']]
        plylst_use.loc[:, 'num_songs'] = plylst_use['songs_id'].map(len)
        plylst_use.loc[:, 'num_tags'] = plylst_use['tags_id'].map(len)
        plylst_use = plylst_use.set_index('nid')

        plylst_train = plylst_use.iloc[:n_train, :]
        plylst_test = plylst_use.iloc[n_train:, :]

        row = np.repeat(range(n_train), plylst_train['num_songs'])
        col = [song for songs in plylst_train['songs_id'] for song in songs]
        dat = np.repeat(1, plylst_train['num_songs'].sum())
        train_songs_A = spr.csr_matrix((dat, (row, col)), shape=(n_train, n_songs))

        row = np.repeat(range(n_train), plylst_train['num_tags'])
        col = [tag for tags in plylst_train['tags_id'] for tag in tags]
        dat = np.repeat(1, plylst_train['num_tags'].sum())
        train_tags_A = spr.csr_matrix((dat, (row, col)), shape=(n_train, n_tags))

        return plylst_train, plylst_test, train_songs_A, train_tags_A, song_dict, tag_dict, plylst_nid_id, song_sid_id, tag_tid_id


    def predict(self, train_df, test_df, train_songs_mat, train_tags_mat, song_dic, tag_dic, plylst_nid_id_dic, song_sid_id_dic, tag_tid_id_dic):

        train = train_df
        test = test_df
        train_songs_A = train_songs_mat
        train_tags_A = train_tags_mat
        song_dict = song_dic
        tag_dict = tag_dic
        plylst_nid_id = plylst_nid_id_dic
        song_sid_id = song_sid_id_dic
        tag_tid_id = tag_tid_id_dic

        train_songs_A_T = train_songs_A.transpose()
        train_tags_A_T = train_tags_A.transpose()

        desc_occ_song_pid = sorted(song_dict.items(), key=lambda kv: kv[1], reverse=True)
        desc_occ_tag_pid = sorted(tag_dict.items(), key=lambda kv: kv[1], reverse=True)

        top_occ_song_id = [song_id for song_id, value in desc_occ_song_pid[:100]]
        top_occ_tag_id = [tag_id for tag_id, value in desc_occ_tag_pid[:10]]

        res = []

        n_songs = len(song_dict)
        pids = test.index

        for pid in tqdm(pids):
            p = np.zeros((n_songs, 1))
            # 하나의 테스트 플레이리스트 기준 있는 노래에 1을 표기
            p[test.loc[pid, 'songs_id']] = 1
            # user X item 내적 (item X 1) => user X 1 [115071, 1] => [115071, ] => 하나의 플레이리스트 기준 다른 플레이리스트들과의 상관관계 값?
            val = train_songs_A.dot(p).reshape(-1)

            # test의 song 값 test의 tag값
            songs_already = test.loc[pid, "songs_id"]
            tags_already = test.loc[pid, "tags_id"]

            # Item X User 매트릭스와 하나의 플레이리스트를 기준으로 유사한 점수를 나타내는 매트릭스와 내적 => 노래들의 값
            cand_song = train_songs_A_T.dot(val)  # (638336, 1)
            cand_song_idx = cand_song.reshape(-1).argsort()[-150:][::-1]  # 상위 150개 노래 도출

            # 이미 노래가 있는 값을 제거하고 상위 100개
            cand_song_idx = cand_song_idx[np.isin(cand_song_idx, songs_already) == False][:100]
            # 기존 ID 값 converting 하여 결과에 담기
            rec_song_idx = [song_sid_id[i] for i in cand_song_idx]

            cand_tag = train_tags_A_T.dot(val)
            cand_tag_idx = cand_tag.reshape(-1).argsort()[-15:][::-1]

            cand_tag_idx = cand_tag_idx[np.isin(cand_tag_idx, tags_already) == False][:10]
            rec_tag_idx = [tag_tid_id[i] for i in cand_tag_idx]

            if len(rec_song_idx) != 100:
                lack_num = 100 - len(rec_song_idx)
                # target_songs = list(set(top_occ_song_id).difference(set(rec_song_idx)))
                target_songs = [song_id for song_id in top_occ_song_id if song_id not in rec_song_idx]
                song_reminder = target_songs[:lack_num]
                rec_song_idx = rec_song_idx + song_reminder

            if len(rec_tag_idx) != 10:
                lack_num = 10 - len(rec_tag_idx)
                # target_tags = list(set(top_occ_tag_id).difference(set(rec_tag_idx)))
                target_tags = [tag_id for tag_id in top_occ_tag_id if tag_id not in rec_tag_idx]
                tag_reminder = target_tags[:lack_num]
                # tag_reminder = top_occ_tag_id[:lack_num]
                rec_tag_idx = rec_tag_idx + tag_reminder

            res.append({
                "id": plylst_nid_id[pid],
                "songs": rec_song_idx,
                "tags": rec_tag_idx
            })

        return res

#class NeuMF:

#class CBF_A1:

#class CBF_A2: