def load_data(type):
    print('loading data...')
    dirs = './data'
    filename = os.path.join(dirs,'sort-of-clevr.pickle')
    f = open(filename, 'r')
    train_datasets, test_datasets = pickle.load(f)
    rel_train = []
    rel_test = []
    norel_train = []
    norel_test = []
    print('processing data...')

    for img, relations, norelations in train_datasets:
        # img = np.swapaxes(img,0,2)
        for qst,ans in zip(relations[0], relations[1]):
            rel_train.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_train.append((img,qst,ans))

    for img, relations, norelations in test_datasets:
        # img = np.swapaxes(img,0,2)
        for qst,ans in zip(relations[0], relations[1]):
            rel_test.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_test.append((img,qst,ans))
    
    if type=="train":
        return (rel_train, nonrel_train)
    else:
        return (rel_test, nonrel_test)
    # return (rel_train, rel_test, norel_train, norel_test)

def cvt_data_axis(data):
    img = [e[0] for e in data]
    qst = [e[1] for e in data]
    ans = [e[2] for e in data]
    return (img,qst,ans)

def tensor_data(data, i, bs):
    img = np.asarray(data[0][bs*i:bs*(i+1)])
    qst = np.asarray(data[1][bs*i:bs*(i+1)])
    ans = np.asarray(data[2][bs*i:bs*(i+1)])
    return (img, qst, ans)
