import pandas as pd
import os

from glob import glob


def read_datadir(path_to_dir, file_ext='tsv', delimiter='\t', **kwargs):
    if not os.path.isdir(path_to_dir):
        raise ValueError("%s is not a directory" % path_to_dir)
    
    name_regex = '*.%s' % file_ext
    data_file_names = glob(os.path.join(path_to_dir, name_regex))
    list_of_dfs = [None]*len(data_file_names)

    for i, fname in enumerate(data_file_names):
        print('%d/%d: %s' % (i+1, len(data_file_names), fname))
        df = pd.read_csv(fname, delimiter=delimiter)
        list_of_dfs[i] = get_events_with_tracklen_equal_to(df, **kwargs)

    return concat_multiple_dataframes(list_of_dfs)


def get_events_with_tracklen_equal_to(df, tracklen=6, fake_id=-1):
    df_no_fakes = df[df.track != fake_id]
    gp_size = df_no_fakes.groupby(['event', 'track']).size()
    # exclude events with tracks which len doesn't equal to tracklen
    gp = gp_size[gp_size != tracklen]
    event_ids = gp.index.get_level_values('event').unique()
    # exclude selected events
    return df[~df.event.isin(event_ids)]


def concat_multiple_dataframes(list_of_dfs):
    print("Concatenate %d datasets" % len(list_of_dfs))
    n_events = 0
    for i in range(len(list_of_dfs)):
        print('%d/%d: Number of events - %d' % 
                (i+1, len(list_of_dfs), list_of_dfs[i].event.nunique()))
        new_event_ids, uniq_events = pd.factorize(list_of_dfs[i].event)
        list_of_dfs[i].loc[:, 'event'] = new_event_ids + n_events
        n_events += len(uniq_events)

    all_df = pd.concat(list_of_dfs)
    assert all_df.event.nunique() == n_events
    print('Total number of events - %d' % all_df.event.nunique())
    return all_df
    