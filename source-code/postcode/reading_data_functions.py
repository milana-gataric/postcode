import numpy as np
import pandas as pd
from pandas import read_csv
from scipy.io import loadmat # to load matlab data


# auxiliary functions required for reading and handling the data
def barcodes_01_from_channels_1234(barcodes_1234, C, R):
    K = barcodes_1234.shape[0]
    barcodes_01 = np.ones((K, C, R))
    for b in range(K):
        barcodes_01[b, :, :] = 1 * np.transpose(barcodes_1234[b, :].reshape(R, 1) == np.arange(1, C + 1))
    return barcodes_01


def barcodes_01_from_letters(barcodes_AGCT, barcode_letters, R):
    K = len(barcodes_AGCT)
    C = len(barcode_letters)
    barcodes_1234 = np.zeros((K, R))
    for k in range(K):
        for r in range(R):
            barcodes_1234[k, r] = np.where(barcode_letters == barcodes_AGCT[k][r])[0][0] + 1
    barcodes_01 = barcodes_01_from_channels_1234(barcodes_1234, C, R)
    return barcodes_01


def read_taglist_and_channel_info(data_path):
    # reads taglist.csv and channel_info.csv and
    # returns barcodes_01 which is a numpy array with 01 entries and dimension K x C x R
    taglist = read_csv(data_path + 'taglist.csv')
    channel_info = read_csv(data_path + 'channel_info.csv')
    gene_names = np.array(taglist.Name)
    barcodes_AGCT = np.array(taglist.Code)

    K = len(taglist)  # number of barcodes
    R = channel_info.nCycles[0]  # number of rounds
    C_total = channel_info.nChannel[0]

    channel_base = []
    coding_chs = []
    channel_names = []
    for i in range(C_total):
        name = channel_info.columns[2 + i]
        base = channel_info.iloc[:, 2 + i][0]
        coding_chs.append(len(base) == 1)
        channel_base.append(base)
        channel_names.append(name)

    C = sum(coding_chs)  # number of coding channels
    barcode_letters = np.array(channel_base)[np.array(coding_chs)]
    barcodes_01 = barcodes_01_from_letters(barcodes_AGCT, barcode_letters, R)

    channels_info = dict()
    for key in ['barcodes_AGCT', 'coding_chs', 'channel_base', 'channel_names']:
        channels_info[key] = locals()[key]
    return barcodes_01, K, R, C, gene_names, channels_info


def read_taglist_and_channel_info_breastdata(data_path):
    taglist = read_csv(data_path + 'taglist.csv')
    channel_info = read_csv(data_path + 'channel_info.csv')
    gene_names = np.array(taglist.Gene)
    barcodes_AGCT = np.array(taglist.Channel)

    K = len(taglist)  # number of barcodes
    R = channel_info.nCycles[0]  # number of rounds
    C_total = channel_info.nChannel[0]

    channel_base = []
    coding_chs = []
    channel_names = []
    for i in range(C_total):
        name = channel_info.iloc[:, 5 + i][0]
        base = channel_info.filter(regex=name).iloc[0, 0]
        coding_chs.append(len(base) == 1)
        channel_base.append(base)
        channel_names.append(name)

    C = sum(coding_chs)  # number of coding channels
    barcode_letters = np.array(channel_base)[np.array(coding_chs)]
    barcodes_01 = barcodes_01_from_letters(barcodes_AGCT, barcode_letters, R)
    channels_info = dict()
    for key in ['barcodes_AGCT', 'coding_chs', 'channel_base', 'channel_names']:
        channels_info[key] = locals()[key]
    return barcodes_01, K, R, C, gene_names, channels_info


def collect_spots_from_mat_files(extracted_spots_path, C, R, tiles_to_load, tile_names, tile_size):
    spots = np.empty((0, C, R))
    # spots = torch.empty((0,C+1,R+1))#Mut-e2
    # spots = torch.empty((0,C+1,R))#Mut-l2,l3
    spots_loc = pd.DataFrame(columns=['X', 'Y', 'Tile'])
    for y_ind in range(tiles_to_load['y_start'], tiles_to_load['y_end'] + 1):  # range(3,4):#
        for x_ind in range(tiles_to_load['x_start'], tiles_to_load['x_end'] + 1):  # range(2,3):#
            tile_name = 'X' + str(x_ind) + '_Y' + str(y_ind)
            if np.isin(tile_name, tile_names['selected_tile_names']):
                # collecting extracted spots needed for decoding:
                extracted_spots = loadmat(extracted_spots_path + 'tile_' + tile_name + '.mat', squeeze_me=True)
                try:
                    spots_i = extracted_spots['spot_intensities_max']  # one tile
                except:
                    spots_i = extracted_spots['spot_intensities_9pix']
                # spots_i = extracted_spots['spot_intensities_mean'] # one tile
                N_i = spots_i.shape[0]
                if N_i > 0:
                    spots = np.concatenate((spots, spots_i))
                    # saving spots locations in a data frame:
                    X = (x_ind - tiles_to_load['x_start']) * tile_size + extracted_spots['centers'][:, 0]
                    Y = (y_ind - tiles_to_load['y_start']) * tile_size + extracted_spots['centers'][:, 1]
                    Tile = np.tile(np.array([tile_name]), N_i)
                    spots_loc_i = pd.DataFrame(
                        np.concatenate((X.reshape((N_i, 1)), Y.reshape((N_i, 1)), Tile.reshape((N_i, 1))), axis=1),
                        columns=['X', 'Y', 'Tile'], index=None)
                    spots_loc = spots_loc.append(spots_loc_i, ignore_index=True)
    return spots, spots_loc

