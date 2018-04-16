import warnings
import numpy
import os
import os.path
import io
import zipfile
import scipy.sparse
import struct
import array

def read_float_array(f, n):
    a = array.array('f')
    a.fromfile(f, n)
    return numpy.fromiter(a, 'float32')

def read_float_array2(f, n):
    fmt = "<%df" % n
    return numpy.array(struct.unpack(fmt, f.read(4*n)))

def write_float_array2(f, v):
    fmt='f'*len(v)
    bin=struct.pack(fmt,*v)
    f.write(bin)

def read_int_array2(f, n):
    fmt = "<%di" % n
    return numpy.array(struct.unpack(fmt, f.read(4*n)))

def write_int_array2(f, v):
    fmt='i'*len(v)
    bin=struct.pack(fmt,*v)
    f.write(bin)

def read_uchar_array2(f, n):
    fmt = "<%dB" % n
    return numpy.array(struct.unpack(fmt, f.read(4*n)))

def write_uchar_array2(f, v):
    fmt='B'*len(v)
    bin=struct.pack(fmt,*v)
    f.write(bin)

def read_float(f):
    return struct.unpack('f', f.read(4))[0]

def read_int(f):
    return struct.unpack('i', f.read(4))[0]

def write_int(f, v):
    f.write(struct.pack('i', v))

def read_uint(f):
    return struct.unpack('I', f.read(4))[0]

def write_uint(f, v):
    f.write(struct.pack('I', v))

def write_float(f, v):
    f.write(struct.pack('f', v))

def read_uchar(f):
    return struct.unpack('B', f.read(1))[0]

def write_uchar(f, v):
    return f.write(struct.pack('B', v))

def _get_label_file(filename):
    f = os.path.basename(filename)
    f = os.path.splitext(f)[0]
    return int(f[:f.find('_')])

def _get_id_file(filename):
    f = os.path.basename(filename)
    f = os.path.splitext(f)[0]
    try:
        return int(f[f.find('_') + 1:])
    except:
        print('Error get id from file:', filename)

def _read_old_dataset(filename):
    zip = zipfile.ZipFile(filename)
    f = zip.open('info.data')

    nsamples = read_int(f)
    nfeats   = read_int(f)
    nclasses = read_int(f)
    nlabels  = read_int(f)
    function = read_int(f)

    id = numpy.zeros(nsamples, dtype=int)
    truelabel = numpy.zeros(nsamples, dtype=int)
    label = numpy.zeros(nsamples, dtype=int)
    weight = numpy.zeros(nsamples)
    status = numpy.zeros(nsamples, dtype=int)
    feats = numpy.zeros((nsamples, nfeats))

    for s in range(nsamples):
        id[s] = read_int(f)
        truelabel[s] = read_int(f)
        label[s] = read_int(f)

        weight[s] = read_float(f)
        status[s] = read_uchar(f)

        feats[s, :] = read_float_array2(f, nfeats)

    f.close()

    ref_data = None

    if 'ref_data.csv' in zip.namelist():
        f = zip.open('ref_data.csv')
        ref_data = numpy.loadtxt(f, dtype='str')



    return dict(id = id, feats = feats, label = label, truelabel = truelabel, status = status, weight = weight, function = function, ref_data = ref_data)

def _change_ref_data_basename(dataset, folder):

    ref_data = dataset['ref_data']
    nfiles = ref_data.shape[0]
    files = [None] * nfiles

    for f in range(nfiles):
        files[f] = ref_data[f].split('/')

    files = numpy.array(files)

    common = (files == files[0,:]).all(axis=0)

    cur_basepath = files[0,common]
    cur_basepath = '/'.join(cur_basepath)

    for f in range(nfiles):
        ref_data[f] = ref_data[f].replace(cur_basepath, folder)

def _save_old_dataset(filename, data):

    feats = data['feats']
    id = data['id']
    truelabel = data['truelabel'].flatten()
    label = data['label']
    weight = data['weight']
    status = data['status']

    nsamples, nfeats = feats.shape

    nclasses = len(numpy.unique(truelabel))
    nlabels = len(numpy.unique(label))

    function = data['function']


    zip = zipfile.ZipFile(filename, mode='w')
    f = io.BytesIO()

    write_int(f, nsamples)
    write_int(f, nfeats)
    write_int(f, nclasses)
    write_int(f, nlabels)
    write_int(f, function)

    for s in range(nsamples):
        write_int(f, id[s])
        write_int(f, truelabel[s])
        write_int(f, label[s])

        write_float(f, weight[s])
        write_uchar(f, status[s])
        # import pdb; pdb.set_trace()
        if scipy.sparse.issparse(feats[s, :]):
            feats_dense = feats[s,:].toarray()
        else:
            feats_dense = feats[s,:]
        write_float_array2(f, feats_dense.flatten())

    # import pdb; pdb.set_trace()
    ref_data = data.get('ref_data', [])

    ref_data_type = 0

    if len(ref_data) > 0:
        ref_data_type = 4

    write_float_array2(f, numpy.ones(nfeats))#alpha
    write_int(f, ref_data_type)#ref data type

    zip.writestr('info.data', f.getvalue())

    header_info = "nfeats: 0\nhas_mean: 0\nhas_stdev: 0\nncomps: 0\nhas_w: 0\nhas_rotation_matrix: 0\nhas_whitening_matrix: 0"
    zip.writestr('feat_space.fsp', header_info)

    if  len(ref_data) > 0:
        zip.writestr('ref_data.csv', '\n'.join(ref_data.tolist()))

    zip.close()

def create_dataset(nsamples, nfeats):
    dataset = dict(version = 1,
                   id = numpy.arange(nsamples).reshape((nsamples, 1)),
                   feats = numpy.zeros((nsamples,nfeats)),
                   projection = numpy.zeros((nsamples,nfeats)),
                   alpha = numpy.ones((1, nfeats)),
                   label = numpy.zeros((nsamples, 1)),
                   truelabel = numpy.zeros((nsamples, 1)),
                   status = numpy.zeros((nsamples, 1)),
                   weight = numpy.ones((nsamples, 1)),
                   nChecked = 0, clusterId = numpy.zeros((nsamples, 1)),
                   nLabels = 0,
                   isSupervised = numpy.zeros((nsamples, 1)),
                   isLabelPropagated = numpy.zeros((nsamples, 1)),
                   function = 1, ref_data = None,
                   nClasses = 0, nTrainSamples = 0,
                   ref_data_type = 0)

    out = IftDataSet()
    out._dataset = dataset
    return out

def _select_dataset(dataset, mask):
    new_data = {}
    # import pdb; pdb.set_trace()

    new_data['projection'] = None
    if dataset['projection'] is not None:
        new_data['projection']= dataset['projection'][mask]
    new_data['isLabelPropagated'] = dataset['isLabelPropagated']
    new_data['nChecked'] = dataset['nChecked']
    new_data['function'] = dataset['function']
    new_data['weight']   = dataset['weight'][mask]
    new_data['status']   = dataset['status'][mask]
    new_data['label']    = dataset['label'][mask]
    new_data['truelabel']= dataset['truelabel'][mask]
    new_data['feats']    = dataset['feats'][mask, :]
    new_data['id']       = dataset['id'][mask]
    new_data['isSupervised'] = dataset['isSupervised'][mask]
    new_data['clusterId'] = dataset['clusterId'][mask]
    new_data['version']  = dataset['version']
    new_data['nClasses'] = dataset['nClasses']
    new_data['alpha']    = dataset['alpha']
    new_data['nLabels']  = dataset['nLabels']
    new_data['ref_data_type'] = dataset['ref_data_type']

    if len(dataset.get('ref_data', []))>0:
        new_data['ref_data'] = dataset['ref_data']#we copy the whole ref_data always

    return new_data

def _read_dataset(filename):
    zip = zipfile.ZipFile(filename)

    version = int(zip.open('version.data').readline())
    feats = _read_float_array_field(zip, 'feat.data')
    projection = _read_float_array_field(zip, 'projection.data')
    alpha = _read_float_array_field(zip, 'alpha.data')
    truelabel = _read_int_array_field(zip, 'true_label.data')
    label = _read_int_array_field(zip, 'label.data')
    id = _read_int_array_field(zip, 'id.data')
    nChecked = _read_int_array_field(zip, 'n_checked.data')
    clusterId = _read_int_array_field(zip, 'group.data')
    status = _read_uchar_array_field(zip, 'status.data')
    isSupervised = _read_uchar_array_field(zip, 'is_supervised.data')
    isLabelPropagated = _read_uchar_array_field(zip, 'is_label_propagated.data')
    weight = _read_float_array_field(zip, 'weight.data')
    nClasses = _read_int_array_field(zip, 'n_classes.data')
    function = _read_int_array_field(zip, 'function_number.data')
    nTrainSamples = _read_int_array_field(zip, 'n_train_samples.data')
    nLabels = _read_int_array_field(zip, 'n_labels.data')
    ref_data_type = _read_int_array_field(zip, 'ref_data_type.data')

    ref_data = None

    if 'ref_data.csv' in zip.namelist():
        f = zip.open('ref_data.csv')
        ref_data = numpy.loadtxt(f, dtype='str')

    return dict(version = version,
        id = id, feats = feats,
        projection = projection, alpha = alpha,
        label = label, truelabel = truelabel,
        status = status, weight = weight,
        nChecked = nChecked, clusterId = clusterId,
        nLabels = nLabels,
        isSupervised = isSupervised,
        isLabelPropagated = isLabelPropagated,
        function = function, ref_data = ref_data,
        nClasses = nClasses, nTrainSamples = nTrainSamples,
        ref_data_type = ref_data_type)


def _save_dataset(filename, dataset):
    zip = zipfile.ZipFile(filename, mode='w')

    zip.writestr('version.data', str(dataset['version']))
    _write_float_array_field(zip, 'feat.data', dataset['feats'])
    _write_float_array_field(zip, 'projection.data', dataset['projection'])
    _write_float_array_field(zip, 'alpha.data', dataset['alpha'])
    _write_int_array_field(zip, 'true_label.data', dataset['truelabel'])
    _write_int_array_field(zip, 'label.data', dataset['label'])
    _write_int_array_field(zip, 'id.data', dataset['id'])
    _write_int_array_field(zip, 'n_checked.data', dataset['nChecked'])
    _write_int_array_field(zip, 'group.data', dataset['clusterId'])
    _write_uchar_array_field(zip, 'status.data', dataset['status'])
    _write_uchar_array_field(zip, 'is_supervised.data', dataset['isSupervised'])
    _write_uchar_array_field(zip, 'is_label_propagated.data', dataset['isLabelPropagated'])
    _write_float_array_field(zip, 'weight.data', dataset['weight'])
    _write_int_array_field(zip, 'n_classes.data', dataset['nClasses'])
    _write_int_array_field(zip, 'function_number.data', dataset['function'])
    _write_int_array_field(zip, 'n_train_samples.data', dataset['nLabels'])
    _write_int_array_field(zip, 'n_labels.data', dataset['nLabels'])
    _write_int_array_field(zip, 'ref_data_type.data', dataset['ref_data_type'])

    ref_data = dataset['ref_data']
    if  dataset['ref_data'] is not None:
        if dataset['ref_data_type']==4:#fileset
            zip.writestr('ref_data.csv', '\n'.join(ref_data.tolist()))

    zip.close()

def _read_h5(filename):
    dataset = deepdish.io.load(filename)
    feats   = dataset['feats']

    nsamples, nfeats = feats.shape

    weight  = dataset.get('weight', numpy.zeros(nsamples))
    status  = dataset.get('status', numpy.zeros(nsamples))
    label   = dataset.get('label', numpy.zeros(nsamples))
    truelabel=dataset.get('truelabel', numpy.zeros(nsamples))
    function = dataset.get('function', 1)
    id = dataset.get('id', numpy.arange(nsamples))

    return dict(feats = feats, label = label, truelabel = truelabel, status = status, weight = weight, id = id, function = function)

def _read_masters_h5(filename):
    dataset = deepdish.io.load(filename)
    feats   = dataset['X']
    truelabel=dataset['Y']

    if(truelabel.min()==0):
        truelabel += 1

    nsamples, nfeats = feats.shape

    weight  = numpy.zeros(nsamples)
    status  = numpy.zeros(nsamples)
    label   = numpy.zeros(nsamples)
    id = numpy.arange(nsamples)

    return dict(feats = feats, label = label, truelabel = truelabel, status = status, weight = weight, id = id, function = 1)

def _read_float_array_field(z, filename):
    try:
        f = z.open(filename)
    except:
        return None
    dim = numpy.fromstring(f.readline(), sep=',').astype(int)
    n = numpy.prod(dim)
    data = read_float_array2(f,n).reshape(dim)
    if n==1:
        return numpy.asscalar(data)
    else:
        return data.reshape(dim)

def _read_uchar_array_field(z, filename):
    try:
        f = z.open(filename)
    except:
        return None
    dim = numpy.fromstring(f.readline(), sep=',').astype(int)
    n = numpy.prod(dim)
    data = read_uchar_array2(f,n).reshape(dim)
    if n==1:
        return numpy.asscalar(data)
    else:
        return data.reshape(dim)

def _read_int_array_field(z, filename):
    try:
        f = z.open(filename)
    except:
        return None
    dim = numpy.fromstring(f.readline(), sep=',').astype(int)
    n = numpy.prod(dim)
    data = read_int_array2(f, n)
    if n==1:
        return numpy.asscalar(data)
    else:
        return data.reshape(dim)

def _write_float_array_field(z, filename, data):
    if data is None:
        return
    if(numpy.isscalar(data)):
        dim = (1,1)
        data = numpy.array(data)
    else:
        dim = data.shape
    f = io.BytesIO()
    dimstr = ','.join(['%d'%x for x in dim])
    f.write(dimstr+'\n')
    n = numpy.prod(dim)
    write_float_array2(f, data.ravel())
    z.writestr(filename, f.getvalue())

def _write_uchar_array_field(z, filename, data):
    if data is None:
        return
    if(numpy.isscalar(data)):
        dim = (1,1)
        data = numpy.array(data)
    else:
        dim = data.shape
    f = io.BytesIO()
    dimstr = ','.join(['%d'%x for x in dim])
    f.write(dimstr+'\n')
    n = numpy.prod(dim)
    write_uchar_array2(f, data.ravel())
    z.writestr(filename, f.getvalue())


def _write_int_array_field(z, filename, data):
    if data is None:
        return
    if(numpy.isscalar(data)):
        dim = (1,1)
        data = numpy.array(data)
    else:
        dim = data.shape
    f = io.BytesIO()
    dimstr = ','.join(['%d'%x for x in dim])
    f.write(dimstr+'\n')
    n = numpy.prod(dim)
    write_int_array2(f, data.ravel())
    z.writestr(filename, f.getvalue())

def read_dataset(filename):
    dataset = IftDataSet()
    try:
        dataset._dataset = _read_dataset(filename)
    except:
        warnings.warn('This dataset is in an obsolete format.\n Relics of an ancient time.\n I am not responsible for anything weird happening.', DeprecationWarning)
        dataset._dataset = _read_old_dataset(filename)
    return dataset

def save_dataset(filename, dataset):
    _save_dataset(filename, dataset._dataset)


class IftDataSet(object):
    def __init__(self):
        self._dataset = {}
        self._filename = None

    def nsamples(self):
        return self._dataset['feats'].shape[0]

    def nfeats(self):
        return self._dataset['feats'].shape[1]

    def save(self, filename):
        save_dataset(filename, self)

    def set_ref_data_basename(self, path):
        _change_ref_data_basename(self._dataset, path)

    def plot(self, title = 'The amazing peixinho\'s plots'):

        import ggplot
        import sklearn.manifold
        import pandas

        reduc = self._dataset.get('projection', None)
        if reduc is None or reduc.shape[1]>=3:
            tsne = sklearn.manifold.TSNE()
            reduc = tsne.fit_transform(self._dataset['feats'])
            self._dataset['projection'] = reduc

        df = pandas.DataFrame({'X': reduc[:,0].ravel(), 'Y':reduc[:,1].ravel(), 'truelabel':self._dataset['truelabel'].ravel()})

        df['truelabel'] = df['truelabel'].astype(object)

        return ggplot.ggplot(ggplot.aes(x='X', y='Y', color='truelabel'), data=df) + ggplot.geom_point() + ggplot.ggtitle(title)


    def __getitem__(self, key):
        if not isinstance(key, (list, tuple)):
            if key not in self._dataset.keys():
                raise IndexError("%s is not a valid IftDataSet field"%key)
            else:
                return self._dataset[key]
        if len(key)>=3:
            raise IndexError("Too many indices for dataset indexing")

        n_samples = self.nsamples()
        n_feats   = self.nfeats()
        dataset = IftDataSet()
        for field in self._dataset.keys():
            idx1 = slice(None, None, None)
            idx2 = slice(None, None, None)
            try:
                shape = self._dataset[field].shape
                if n_samples in shape:
                    idx1 = key[0]
                if n_feats in shape:
                    idx2 = key[1]

                if isinstance(idx1, (list, tuple)):
                    idx1 = numpy.array(idx1)
                if isinstance(idx2, (list, tuple)):
                    idx2 = numpy.array(idx2)

                if isinstance(idx1, (numpy.ndarray, numpy.generic) ):
                    if idx1.dtype == bool:
                        idx1 = numpy.where(idx1)[0]
                if isinstance(idx2, (numpy.ndarray, numpy.generic) ):
                    if idx2.dtype == bool:
                        idx2 = numpy.where(idx2)[1]


                ndims = len(self._dataset[field])

                # if field=='feats':
                #     import pdb; pdb.set_trace()
                if isinstance(idx1, (numpy.ndarray, numpy.generic) ) and isinstance(idx2, (numpy.ndarray, numpy.generic) ):
                    ixgrid = numpy.ix_(idx1,idx2)
                    dataset._dataset[field] = self._dataset[field][ixgrid]
                else:
                    dataset._dataset[field] = self._dataset[field][idx1, idx2]

            except Exception as e:
                dataset._dataset[field] = self._dataset[field]
        return dataset

    def __setitem__(self, field, val):
        try:
            self._dataset[field] = val
        except:
            'There is no %s in dataset'%field

    def save(self, filename = None):
        if filename == None:
            filename = self._filename

        if filename!=None:
            save_dataset(filename, self)
            self._filename = filename
        else:
            raise 'Please inform filepath to save the dataset'
