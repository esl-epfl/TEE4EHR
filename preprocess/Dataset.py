import numpy as np
import torch
import torch.utils.data

from transformer import Constants
from torch.utils.data import WeightedRandomSampler






class TEDA(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, data_event, dict_state=None, dim='MHP', data_label='multiclass', have_label=False, have_demo=False):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """

        self.have_label=have_label
        self.have_demo=have_demo
        self.have_state = False # will change later

        self.time = [[elem['time_since_start'] for elem in inst] for inst in data_event]
        self.time_gap = [[elem['time_since_last_event'] for elem in inst] for inst in data_event]
        # plus 1 since there could be event type 0, but we use 0 as padding
        

        if isinstance(data_event[0][0]['type_event'],np.ndarray):
            self.event_type = [[elem['type_event'] + 0 for elem in inst] for inst in data_event]
        else:
            self.event_type = [[elem['type_event'] + 1 for elem in inst] for inst in data_event]

        self.length = len(data_event)



        if dict_state is not None:
            self.have_state=True

            self.time_state = [[elem['abs_time'] for elem in inst] for inst in dict_state['state']]
            self.value = [[elem['value'] for elem in inst] for inst in dict_state['state']]
            self.mod = [[elem['mod']+1 for elem in inst] for inst in dict_state['state']]        
            

        if self.have_label:        
            self.label = [[elem['label'] for elem in inst] for inst in dict_state['state']]
            self.whole_label = [(sum([elem['label'] for elem in inst])>0)+0 for inst in dict_state['state']]


        if self.have_demo:        
            self.demo = [inst for inst in dict_state['demo']]


        a=1

    def __len__(self):
        return self.length
    def sample_label(self):
        return self.whole_label
    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        # print('get')
        sample={'time':self.time[idx], 'time_gap':self.time_gap[idx], 'event_type':self.event_type[idx],
        }

        if self.have_state:
            sample.update({
                'time_state':self.time_state[idx], 'value':self.value[idx], 'mod':self.mod[idx],
            })

        if self.have_demo:
            sample.update({   'demo':self.demo[idx]   })

        if self.have_label:
            sample.update({   'label':self.label[idx]   })

        
        return sample










def pad_time(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)
    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])


    return torch.tensor(batch_seq, dtype=torch.float32)


def pad_type(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    
    # if multilabel:
    if isinstance(insts[0][0],np.ndarray):
        vec_len = len(insts[0][0])
        batch_seq = np.stack([
            np.concatenate([
                inst,
                np.zeros((   max_len - len(inst) , len(inst[0])))
            ])
            for inst in insts
        ]) #[B,L,K] K dimension of encoding
    else: # then multiclass
        batch_seq = np.array([
            inst + [Constants.PAD] * (max_len - len(inst))
            for inst in insts])


    return torch.tensor(batch_seq, dtype=torch.long)



def collate_fn(insts):
    """ Collate function, as required by PyTorch. """
    
    # time, time_gap, event_type, time_state, value, mod, label = list(zip(*insts))
    
    time = [ inst['time'] for inst in insts]
    time_gap = [ inst['time_gap'] for inst in insts]
    event_type = [ inst['event_type'] for inst in insts]

    time = pad_time(time) # [B,L]
    time_gap = pad_time(time_gap) # [B,L]
    event_type = pad_type(event_type) # [B,L,K]

    out=[time, time_gap, event_type]

    if 'time_state' in insts[0]:

        time_state = [ inst['time_state'] for inst in insts]
        value = [ inst['value'] for inst in insts]
        mod = [ inst['mod'] for inst in insts]

        time_state = pad_time(time_state) # [B,P]
        value = pad_time(value) # [B,P]
        mod= pad_type(mod) # [B,P]
        out.extend([time_state, value, mod])

    if 'label' in insts[0]:
        label = [ inst['label'] for inst in insts]
        label= pad_type(label) # [B,P]
        out.append(label)

    
    if 'demo' in insts[0]:
        demo = torch.tensor( [ inst['demo'] for inst in insts] ) # [B, num_demos]
        out.append(demo)

    return out



def get_dataloader(data_event, data_state=None, bs=4, shuffle=True, dim='MHP', data_label='multiclass', balanced=False, state_args=None):
    """ Prepare dataloader. """

    ds = TEDA(data_event, data_state,dim=dim, data_label=data_label,**state_args)
    

    if balanced and hasattr(ds, 'whole_label'):
        
        sample_labels = ds.sample_label()
        pos_count = sum(sample_labels)
        neg_count = len(sample_labels) - sum(sample_labels)
        class_counts = [neg_count, pos_count]
        sample_weights = [1/class_counts[i] for i in sample_labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(ds), replacement=True)

        print(f'[info] True/total = {pos_count/(pos_count+neg_count) :.4f}')
        
        print(f'[info] balanced mini batches')

        dl = torch.utils.data.DataLoader(
            ds,
            num_workers=0,
            batch_size=bs,
            collate_fn=collate_fn,
            # shuffle=shuffle,
            drop_last=True,
            sampler=sampler,
            # pin_memory=True,
        )

    else:
        dl = torch.utils.data.DataLoader(
            ds,
            num_workers=0,
            batch_size=bs,
            collate_fn=collate_fn,
            shuffle=shuffle,
            drop_last=True,
            # sampler=sampler,
            # pin_memory=True,

        )


    return dl


