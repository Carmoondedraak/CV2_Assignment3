from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


# Helper function to quickly see the values of a list or dictionary of data
def printTensorList(data, detailed=False):
    if isinstance(data, dict):
        print('Dictionary Containing: ')
        print('{')
        for key, tensor in data.items():
            print('\t', key, end='')
            print(' with Tensor of Size: ', tensor.size())
            if detailed:
                print('\t\tMin: %0.4f, Mean: %0.4f, Max: %0.4f' % (tensor.min(),
                                                                   tensor.mean(),
                                                                   tensor.max()))
        print('}')
    else:
        print('List Containing: ')
        print('[')
        for tensor in data:
            print('\tTensor of Size: ', tensor.size())
            if detailed:
                print('\t\tMin: %0.4f, Mean: %0.4f, Max: %0.4f' % (tensor.min(),
                                                                   tensor.mean(),
                                                                   tensor.max()))
        print(']')


class SwappedDatasetLoader(Dataset):

    def __init__(self, data_file, prefix, resize=256):
        self.prefix = prefix
        self.resize = resize
        my_file = open(prefix+data_file, "r")
        content = my_file.read()
        self.data_paths = content.split('\n')
        self.transform = transforms.Compose([transforms.Resize((self.resize,self.resize)), transforms.ToTensor(),transforms.Normalize(mean=[0.5],std=[0.5])])
        # self.transform =transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
        # self.dataset = [self.__getitem__(i) for i in range(len(self.data_paths)-1)]
        # Define your initializations and the transforms here. You can also
        # define your tensor transforms to normalize and resize your tensors.
        # As a rule of thumb, put anything that remains constant here.

    def __len__(self):
        # Return the length of the datastructure that is your dataset
        return len(self.data_paths)
    def __getitem__(self, index):
        # Write your data loading logic here. It is much more efficient to
        # return your data modalities as a dictionary rather than list. So you
        # can return something like the follows:
        #     image_dict = {'source': source,
        #                   'target': target,
        #                   'swap': swap,
        #                   'mask': mask}

        # print(self.data_paths[index])
        # print(self.data_paths[index].split('_'))
        x,y,z = self.data_paths[index].split('_')[0],self.data_paths[index].split('_')[2],self.data_paths[index].split('_')[3]
        # print(x,y,z)

        swap = Image.open(self.prefix + 'data/' +self.data_paths[index])
        target = Image.open(self.prefix + 'data/' +x + '_bg_' + y +'.png')
        source = Image.open(self.prefix + 'data/' +x + '_fg_' + z)
        mask = Image.open(self.prefix + 'data/' +x +'_mask_' + y + '_' + z )
        if self.resize:
            swap = self.transform(swap)
            target = self.transform(target)
            source = self.transform(source)
            mask = self.transform(mask)
        image_dict = {'source': source, 'target': target,'swap': swap, 'mask': mask}
        return image_dict, self.data_paths[index]


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import time

    # It is always a good practice to have separate debug section for your
    # functions. Test if your dataloader is working here. This template creates
    # an instance of your dataloader and loads 20 instances from the dataset.
    # Fill in the missing part. This section is only run when the current file
    # is run and ignored when this file is imported.

    # This points to the root of the dataset
    data_root = '../Data/data_set/'
    train_list = 'train.str'
    test_list = 'test.str'
    # This points to a file that contains the list of the filenames to be
    # loaded.
    print('[+] Init dataloader')
    # Fill in your dataset initializations
    testSet = SwappedDatasetLoader(test_list, data_root)
    print('[+] Create workers')
    loader = DataLoader(testSet, batch_size=1, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True)
    print('[*] Dataset size: ', len(loader))
    enu = enumerate(loader)
    for i in range(20):
        a = time.time()
        i, (images) = next(enu)
        b = time.time()
        # Uncomment to use a prettily printed version of a dict returned by the
        # dataloader.
        # printTensorList(images[0], True)
        print('[*] Time taken: ', b - a)
