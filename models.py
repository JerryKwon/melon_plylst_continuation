from collections import Counter
import numpy as np
import pandas as pd
import scipy.sparse as spr
from tqdm.notebook import tqdm

class ICBF_OCC:
    def __init__(self, train_df, test_df):
        self.train = train_df
        self.test = test_df

    def __main__(self):

        plylst_train, plylst_test, train_songs_A, train_tags_A, song_dict, tag_dict, plylst_nid_id, song_sid_id, tag_tid_id = self.clean_b4_train()
        result = self.train(plylst_train,plylst_test,train_songs_A,train_tags_A, song_dict, tag_dict, plylst_nid_id, song_sid_id, tag_tid_id)

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


    def train(self, train_df, test_df, train_songs_mat, train_tags_mat, song_dic, tag_dic, plylst_nid_id_dic, song_sid_id_dic, tag_tid_id_dic):

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