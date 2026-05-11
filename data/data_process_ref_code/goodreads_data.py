import torch
import os.path as op
import torch.utils.data as data

import pandas as pd
import random

class GoodreadsData(data.Dataset):
    def __init__(self, data_dir='./data/goodreads',
                 stage=None,
                 cans_num=20,
                 sep=", ",
                 device=None,
                 no_augment=True):
        self.__dict__.update(locals())
        self.aug = (stage=='train') and not no_augment
        self.padding_item_id=4550
        self.device=device
        self.cans_num = cans_num
        self.check_files()

    def __len__(self):
        return len(self.session_data['seq'])

    def __getitem__(self, i):
        temp = self.session_data.iloc[i]
        candidates = self.negative_sampling(temp['seq_unpad'],temp['next'])
        cans_name=[self.item_id2name[can] for can in candidates]
        seq = torch.tensor(temp['seq']).to(self.device)
        len_seq = torch.tensor(temp['len_seq']).to(self.device)
        len_seq_list = temp['len_seq']
        sample = {
            'seq': seq,
            'movie_seq': temp['movie_seq'],
            'len_seq': len_seq,
            'len_seq_list': len_seq_list,
            'cans': torch.tensor(candidates).to(self.device),
            'cans_name': cans_name,
            'cans_str': '::'.join(cans_name),
            'len_cans': self.cans_num,
            'next_id': torch.tensor(temp['next']).to(self.device),
            'next_title': temp['next_title'],
            'correct_answer': temp['next_title']
        }
        return sample
    
    def negative_sampling(self,seq_unpad,next_item):
        canset=[i for i in list(self.item_id2name.keys()) if i not in seq_unpad and i!=next_item]
        rng = random.Random(next_item)
        candidates=rng.sample(canset, self.cans_num-1)+[next_item]
        rng.shuffle(candidates)
        return candidates  

    def check_files(self):
        self.item_id2name=self.get_game_id2name()
        if self.stage=='train':
            filename="train_data.df"
        elif self.stage=='val':
            filename="val_data.df"
        elif self.stage=='test':
            filename="Test_data.df"
        else:
            raise ValueError(f"Unsupported stage: {self.stage}")
        data_path=op.join(self.data_dir, filename)
        self.session_data = self.session_data4frame(data_path, self.item_id2name)  

        
    def get_game_id2name(self):
        game_id2name = dict()
        item_path=op.join(self.data_dir, 'id2name.txt')
        with open(item_path, 'r') as f:
            for l in f.readlines():
                ll = l.strip('\n').split('::')
                g_name = ll[1].split('(')
                game_id2name[int(ll[0])] = g_name[0].strip()
        return game_id2name
    
    def session_data4frame(self, datapath, game_id2name):
        train_data = pd.read_pickle(datapath)
        train_data = train_data[train_data['len_seq'] >= 3]
        def remove_padding(xx):
            x = xx[:]
            for i in range(10):
                try:
                    x.remove(self.padding_item_id)
                except:
                    break
            return x
        train_data['seq_unpad'] = train_data['seq'].apply(remove_padding)

        def seq_to_title(x): 
            game_list = [self.item_id2name[i] for i in x]
            title_str = '::'.join(game_list)
            return title_str
        
        train_data['movie_seq'] = train_data['seq_unpad'].apply(seq_to_title)


        def id_to_movie(x):
            return self.item_id2name[x]
        train_data['next_title'] = train_data['next'].apply(id_to_movie)
        return train_data




    
