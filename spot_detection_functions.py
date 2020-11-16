import numpy as np
import pandas as pd
import tifffile
import trackpy
from skimage.morphology import white_tophat, disk


# functions for detecting and extracting spots from registered images required prior to decoding
def detect_and_extract_spots(imgs_coding, anchors, C, R, imgs_also_without_tophat=None, compute_also_without_tophat=False,
                             trackpy_diam_detect=5, trackpy_search_range=3):
    ###############
    # detects spots using R images passed via variable 'anchors' and
    # extracts detected spots using CxR images passed via 'imgs_coding'
    ###############
    # apply trackpy to each round
    trackpy.quiet(suppress=True)
    tracks = trackpy.link_df(trackpy.batch(anchors, diameter=trackpy_diam_detect), search_range=trackpy_search_range)
    # find spots appearing in all cycles
    spots_id_all_anchors = tracks['particle'][tracks['frame'] == 0].unique()
    for ind_cy in range(1, R):
        spots_id_all_anchors = np.intersect1d(spots_id_all_anchors,
                                              tracks['particle'][tracks['frame'] == ind_cy].unique())
    num_spots = len(spots_id_all_anchors)
    # print('The number of spots linked across all cycles: {}'.format(num_spots))
    # collect centers of all spots
    centers_y = np.zeros((num_spots, R))
    centers_x = np.zeros((num_spots, R))
    for i in range(num_spots):
        id_particle = spots_id_all_anchors[i]
        centers_y[i, :] = np.array(tracks[tracks['particle'] == id_particle]['y'])
        centers_x[i, :] = np.array(tracks[tracks['particle'] == id_particle]['x'])
    centers_c01 = np.transpose(
        np.stack((centers_x[:, 0] + 1, centers_y[:, 0] + 1)))  # equivalent to the centers in matlab

    # extract max intensity values from each (top-hat-ed) coding channel at the center coordinates +- 1 pixel
    spot_intensities_d = np.zeros((9, num_spots, C, R))
    d = -1
    for dx in np.array([-1, 0, 1]):
        for dy in np.array([-1, 0, 1]):
            d = d + 1
            for ind_cy in range(R):
                x_coord = np.maximum(0, np.minimum(imgs_coding.shape[1] - 1,
                                                   np.around(centers_x[:, ind_cy]).astype('int32') + dx))
                y_coord = np.maximum(0, np.minimum(imgs_coding.shape[0] - 1,
                                                   np.around(centers_y[:, ind_cy]).astype('int32') + dy))
                for ind_ch in range(C):
                    spot_intensities_d[d, :, ind_ch, ind_cy] = imgs_coding[y_coord, x_coord, ind_ch, ind_cy]
    spot_intensities = np.max(spot_intensities_d, axis=0)

    if compute_also_without_tophat:
        spot_intensities_d = np.zeros((9, num_spots, C, R))
        d = -1
        for dx in np.array([-1, 0, 1]):
            for dy in np.array([-1, 0, 1]):
                d = d + 1
                for ind_cy in range(R):
                    x_coord = np.maximum(0, np.minimum(imgs_also_without_tophat.shape[1] - 1,
                                                       np.around(centers_x[:, ind_cy]).astype('int32') + dx))
                    y_coord = np.maximum(0, np.minimum(imgs_also_without_tophat.shape[0] - 1,
                                                       np.around(centers_y[:, ind_cy]).astype('int32') + dy))
                    for ind_ch in range(C):
                        spot_intensities_d[d, :, ind_ch, ind_cy] = imgs_also_without_tophat[y_coord, x_coord, ind_ch, ind_cy]
        spot_intensities_notophat = np.max(spot_intensities_d, axis=0)
    else:
        spot_intensities_notophat = None
    return spot_intensities, centers_c01, spot_intensities_notophat


def load_tiles_to_extract_spots(tifs_path, channels_info, C, R,
                                tile_names, tiles_info, tiles_to_load,
                                spots_params, anchor_available=True, compute_also_without_tophat=False):
    spots = np.empty((0, C, R))
    spots_notophat = np.empty((0, C, R))
    spots_loc = pd.DataFrame(columns=['X', 'Y', 'Tile'])
    print('Extracting spots from: ', end='')
    for y_ind in range(tiles_to_load['y_start'], tiles_to_load['y_end'] + 1):
        for x_ind in range(tiles_to_load['x_start'], tiles_to_load['x_end'] + 1):
            tile_name = 'X' + str(x_ind) + '_Y' + str(y_ind)
            if np.isin(tile_name, tile_names['selected_tile_names']):
                # load selected tile
                print(tile_name, end=' ')
                if x_ind == tiles_info['x_max']:
                    tile_size_x = tiles_info['x_max_size']
                else:
                    tile_size_x = tiles_info['tile_size']
                if y_ind == tiles_info['y_max']:
                    tile_size_y = tiles_info['y_max_size']
                else:
                    tile_size_y = tiles_info['tile_size']

                imgs = np.zeros((tile_size_y, tile_size_x, len(channels_info['channel_names']), R))
                for ind_cy in range(R):
                    for ind_ch in range(len(channels_info['channel_names'])):
                        if channels_info['channel_names'][ind_ch] != 'DAPI':  # no need for dapi
                            try:
                                imgs[:, :, ind_ch, ind_cy] = tifffile.imread(
                                    tifs_path + tiles_info['filename_prefix'] + channels_info['channel_names'][
                                        ind_ch] + '_c0' + str(
                                        ind_cy + 1) + '_' + tile_name + '.tif').astype(np.float32)
                            except:
                                imgs[:, :, ind_ch, ind_cy] = tifffile.imread(
                                    tifs_path + tiles_info['filename_prefix'] + tile_name + '_c0' + str(
                                        ind_cy + 1) + '_' + channels_info['channel_names'][ind_ch] + '.tif').astype(
                                    np.float32)

                imgs_coding = imgs[:, :, np.where(np.array(channels_info['coding_chs']) == True)[0], :]
                # apply top-hat filtering to each coding channel
                imgs_coding_tophat = np.zeros_like(imgs_coding)
                for ind_cy in range(R):
                    for ind_ch in range(C):
                        imgs_coding_tophat[:, :, ind_ch, ind_cy] = white_tophat(imgs_coding[:, :, ind_ch, ind_cy],
                                                                                disk(spots_params['spot_diam_tophat']))
                # extract anchor channel across all cycles
                if anchor_available:
                    anchors = np.swapaxes(
                    np.swapaxes(
                        np.squeeze(imgs[:, :, np.where(np.array(channels_info['channel_base']) == 'anchor')[0][0], :]),
                        0, 2), 1, 2)
                else:
                    # if anchor is not available, form "quasi-anchors" from coding channels
                    anchors = np.swapaxes(np.swapaxes(imgs_coding_tophat.max(axis=2), 0, 2), 1, 2)

                # detect and extract spots from the loaded tile
                spots_i, centers_i, spots_notophat_i = detect_and_extract_spots(imgs_coding_tophat, anchors, C, R,
                                                                                imgs_coding,
                                                                                compute_also_without_tophat,
                                                                                spots_params['trackpy_diam_detect'],
                                                                                spots_params['trackpy_search_range'])
                N_i = spots_i.shape[0]
                if N_i > 0:
                    spots = np.concatenate((spots, spots_i))
                    if compute_also_without_tophat:
                        spots_notophat = np.concatenate((spots_notophat, spots_notophat_i))
                    # saving spots locations from the 1st cycle in a data frame (needed for ploting)
                    X = (x_ind - tiles_to_load['x_start']) * tiles_info['tile_size'] + centers_i[:, 0]
                    Y = (y_ind - tiles_to_load['y_start']) * tiles_info['tile_size'] + centers_i[:, 1]
                    Tile = np.tile(np.array([tile_name]), N_i)
                    spots_loc_i = pd.DataFrame(
                        np.concatenate((X.reshape((N_i, 1)), Y.reshape((N_i, 1)), Tile.reshape((N_i, 1))), axis=1),
                        columns=['X', 'Y', 'Tile'], index=None)
                    spots_loc = spots_loc.append(spots_loc_i, ignore_index=True)
    return spots, spots_loc, spots_notophat


