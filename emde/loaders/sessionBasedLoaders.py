import pandas as pd


def create_sessionbased_dataset(data):
    """
    :param pd.DataFrame data: Data frame with sessions.
            It contains the transactions of the sessions. It must have a header.
            It must have following columns:
                Time - unix timestamp of the events
                SessionId - session IDs
                ItemId - item IDs

    :return pd.Dataframe: Transformed input dataframe to sessions with columns:
            history - all previous user item interaction
            target - the next, predicted item in session
            target_multi - all next items in session
            time_diff - time difference between events in `history`
    """
    rows = []
    session_items = []
    session_timestamps = []
    last_session = -1

    for i in data.index:
        session_id = data.at[i,'SessionId']
        item_id = str(data.at[i,'ItemId'])
        time_stamp = data.at[i,'Time']
        if session_id != last_session:
            if len(session_items) > 1:
                for item_idx in range(1, len(session_items)):
                    rows.append({'history' : ' '.join(session_items[:item_idx]),
                    'target': session_items[item_idx],
                    'target_multi': ' '.join(session_items[item_idx:]),
                    'time_diff': ' '.join(diff_timestamps[:item_idx])})
            session_items = []
            session_timestamps = []
        session_items.append(item_id)

        diff_timestamps = ['0']
        for t in range(1, len(session_timestamps)):
            assert session_timestamps[t] - session_timestamps[t-1] >= 0
            diff_timestamps.append(str(session_timestamps[t] - session_timestamps[t-1]))

        session_timestamps.append(time_stamp)
        last_session = session_id

    if len(session_items) > 1:
        for item_idx in range(1, len(session_items)):
            rows.append({'history' : ' '.join(session_items[:item_idx]),
            'target': session_items[item_idx],
            'target_multi': ' '.join(session_items[item_idx:]),
            'time_diff': ' '.join(diff_timestamps[:item_idx])})
    return pd.DataFrame(rows)