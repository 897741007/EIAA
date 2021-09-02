import pandas as pd
import numpy as np
from pandas.io.parsers import read_csv

class log_rec():
    def __init__(self, log_file):
        self.log_file = log_file
        self.mark = log_file.split('.')[-1]
        self.info_, self.log_ = self.load_log()

    def load_log(self):
        with open(self.log_file, 'r') as f:
            rec = []
            info = []
            tag_r = 0
            tag_i = 0
            for l in f:
                if l.startswith('='):
                    rec.append([])
                    info.append([])
                    tag_r = 0
                    tag_i = 1
                elif l.startswith('-'):
                    tag_r = 1
                    tag_i = 0
                if tag_i:
                    info[-1].append(l.strip())
                if tag_r:
                    rec[-1].append(l.strip())
        return info, rec

    def sep_rec(self, rec):
        rec_ = rec.strip('[]').split('] [')
        rec_arg = rec_[0]
        rec_terms = {}
        for t in rec_[1:]:
            tn, vn = t.split(' ')
            rec_terms[tn] = vn
        return rec_arg, rec_terms

    def get_term(self, arg, terms, terms_type=None, to_file=False):
        if terms_type == None:
            terms_type = [float for _ in terms]
        else:
            assert len(terms) == len(terms_type)
        filtered = []
        for rep in self.log_:
            filtered.append([])
            for rec in rep:
                rec_arg, reg_terms = self.sep_rec(rec)
                if rec_arg == arg:
                    temp = [reg_terms[i] for i in terms]
                    temp = [terms_type[i](temp[i]) for i in range(len(terms))]
                    filtered[-1].append(temp)
        if to_file:
            for rep_idx, rep in enumerate(filtered):
                index = list(range(len(rep)))
                file_data = {'index':index}
                for t_idx in range(len(terms)):
                    file_data[terms[t_idx]] = [i[t_idx] for i in rep]
                file_data = pd.DataFrame(file_data)
                file_data.to_csv('{0}_analysis_rep{1}.csv'.format(self.log_file, rep_idx), index=False)
        else:
            return filtered

    def get_val_test(self, term, by='step', to_file=False):
        assert by in ('step', 'epoch')
        filtered = []
        for rep in self.log_:
            filtered.append({'validation':[], 'test':[]})
            for rec in rep:
                rec_arg, rec_terms = self.sep_rec(rec)
                if ' ' in rec_arg:
                    tag_s, tag_p = rec_arg.split(' ')
                    if tag_s == by:
                        v = float(rec_terms[term])
                        filtered[-1][tag_p].append(v)
        if not to_file:
            store = []
        for rep_idx, rep in enumerate(filtered):
            index = list(range(1, len(rep['test'])+1))
            file_data = {'index':index, 'validation':rep['validation'], 'test':rep['test']}
            file_data = pd.DataFrame(file_data)
            if to_file:
                file_data.to_csv('{0}_{1}_rep{2}.csv'.format(self.mark, term, rep_idx), index=False)
            else:
                store.append(file_data)
        if not to_file:
            return store

def summary(files, best='min'):
    assert best in ('min', 'max')
    summ = {}
    for f in files:
        strategy = '_'.join(f.split('_')[:-1])
        d = pd.read_csv(f)
        val = d['validation']
        test = d['test']
        if best == 'min':
            best_val_loc = np.argmin(val)
        else:
            best_val_loc = np.argmax(val)
        best_test = test[best_val_loc]
        summ[strategy] = best_test
    #summ = pd.DataFrame(summ)
    return summ

def summ(files):
    k = log_rec(files)
    z = k.get_val_test('MAE')
    log = {}
    ilter = len(k.info_)
    tt = []
    val = []
    b_tt = []
    for i in range(ilter-1, -1, -1):
        tt.append(min(z[i]['test']))
        val.append(min(z[i]['validation']))
        b_tt.append(z[i]['test'][np.argmin(z[i]['validation'])])
    log['test'] = tt
    log['validation'] = val
    log['test_at_validation'] = b_tt
    return log
