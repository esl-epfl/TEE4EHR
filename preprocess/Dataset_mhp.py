import numpy as np
import torch
import torch.utils.data
import pickle

from transformer import Constants


class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, data):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """
        # dict_keys(['sid', 'labtimes', 'labtimes_gap', 'labevents', 'stateevents', 'stateevents_converted', 'labtimes_gap_trans', 'num_labevents', 'dict_map_labevents', 'num_stateevents', 'dict_map_stateevents', 'pt'])        self.labtimes = data['labtimes'] # list of N patients, each is a list of L time stampls
        
        # self.stateevents_converted = data['stateevents_converted'] # list of N patients, each is a np.array[L,num_stateevents] 
        self.times = data['times'] # list of N patients, each is a np.array[L,num_stateevents] 

        self.timegaps = data['timegaps'] # list of N patients, each is a list of L time stampls
        self.timegaps_trans = data['timegaps_trans'] # list of N patients, each is a list of L time stampls

        self.events = data['events'] # list of N patients, each is a np.array[L,num_labevents] 
        
        self.pt = data['pt']    # this is the power transformer


        # self.time = [[elem['time_since_start'] for elem in inst] for inst in data]
        # self.time_gap = [[elem['time_since_last_event'] for elem in inst] for inst in data]
        # # plus 1 since there could be event type 0, but we use 0 as padding
        # self.event_type = [[elem['type_event'] + 1 for elem in inst] for inst in data]

        self.length = len(data['id'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        # returns a patient data from the list
        # print(self.pt)
        return self.times[idx],self.timegaps[idx],self.timegaps_trans[idx], self.events[idx], self.pt


def pad_time(insts):
    """ Pad the instance to the max seq length in batch. """
    # input: list[n_batch] each element: is a list[L]

    
    max_len = max([len(inst) for inst in insts])

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32) # shape[B, L]


def pad_type(insts):
    """ Pad the instance to the max seq length in batch. """
    # input: list[n_batch] each element: is a np.array[L, num_types]

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        np.pad(inst[:,None], (  (0,max_len - inst.shape[0]),(0,0)  ), 'constant', constant_values=Constants.PAD ) # inst.shape[0]=L
        for inst in insts])
    # shape[B, L, num_types]

    return torch.tensor(batch_seq, dtype=torch.long).squeeze() # shape[B, L, num_types]


def collate_fn(insts):
    """ Collate function, as required by PyTorch. """
    # insts is a list of size[batch]
    # insts[0] is a list of size 3 (self.labtimes[idx], self.labevents[idx], self.stateevents_converted[idx])
    times, timegaps, timegaps_trans, events, pt = list(zip(*insts)) # each is a list of size n_batch
    # for inst in labtimes_gap_trans:
    #     if not isinstance(inst, list):
    #         aaa=1
    times = pad_time(times) # shape[B, L]
    timegaps = pad_time(timegaps)
    events = pad_type(events) # shape[B, L, num_types]
    timegaps_trans = pad_time(timegaps_trans)

    
    return times, timegaps, events#, timegaps_trans, pt[0]


def get_dataloader(data, batch_size, shuffle=True, num_workers=0):
    """ Prepare dataloader. """

    ds = EventData(data)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )
    return dl





def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            # data.keys()(['sid', 'labtimes', 'labevents', 'stateevents', 'stateevents_converted', 'num_labevents', 'dict_map_labevents', 'num_stateevents', 'dict_map_stateevents'])

            num_types = data['num_types']

            # data = data[dict_name]
            return data, int(num_types), 

    print('[Info] Loading train data...')
    train_data, num_types = load_data(opt.data + 'train.pkl')
    print('[Info] Loading dev data...')
    dev_data, _ = load_data(opt.data + 'dev.pkl')
    print('[Info] Loading test data...')
    test_data, _,= load_data(opt.data + 'test.pkl')


    # set limits for loading
    # per=opt.per
    per=100
    print(f""" 
        train size: {len(train_data)}
        test size: {len(test_data)}
        dev size: {len(dev_data)}

        percentage: {per}
    """)
    # train_data = train_data[:int(per/100 * len(train_data) )]
    

    # test_data = {k:v[:N_sub] if isinstance(v,list) else v for k,v in test_data.items()}
    # dev_data = {k:v[:N_sub] if isinstance(v,list) else v for k,v in dev_data.items()}


    # limit = sum(np.cumsum(np.array([len(x) for x in train_data]))<50000000)
    # train_data = train_data[:limit]
    # limit = sum(np.cumsum(np.array([len(x) for x in dev_data]))<10000000)
    # dev_data = dev_data[:limit]
    # limit = sum(np.cumsum(np.array([len(x) for x in test_data]))<10000000)
    # test_data = test_data[:limit]
    opt.num_workers = 1
    
    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True, num_workers = opt.num_workers)
    testloader = get_dataloader(test_data, opt.batch_size, shuffle=False, num_workers = opt.num_workers)
    return trainloader, testloader, num_types

